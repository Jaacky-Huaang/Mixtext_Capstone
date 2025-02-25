import argparse
import datetime
import os
import json
import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer
import numpy as np
from methods.supervised import run_supervised_experiment
from methods.detectgpt import run_detectgpt_experiments
from methods.gptzero import run_gptzero_experiment
from methods.radar import run_radar
from methods.sentinel import run_sentinel
from methods.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, run_GLTR_experiment
import pdb


# Change the DATA_FILE to the path of the data file
DATA_FILE = "/scratch/jh7956/mixset_data/truncated_data.json"
# python /scratch/jh7956/MixSet/benchmark.py --dataset All --Mixcase_filename gpt4_polish_token.json --seed 0 --mixcase_threshold 1 --train_threshold 10000 --train_with_mixcase --finetune --three_classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="TruthfulQA")
    parser.add_argument('--detectLLM', type=str, default="ChatGPT")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name',
                        type=str, default="t5-large")
    parser.add_argument('--cache_dir', type=str, default=".cache")
    parser.add_argument('--DEVICE', type=str, default="cuda")

    # params for DetectGPT
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbation_list', type=str, default="10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')

    # params for GPTZero
    parser.add_argument('--gptzero_key', type=str, default="")
    
    # params for Mixcase
    parser.add_argument('--Mixcase_filename', type=str, default=None)
    parser.add_argument('--MGT_only_GPT', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--train_threshold', type=int, default=10000)
    parser.add_argument('--test_threshold', type=int, default=2000)
    parser.add_argument('--no_auc', action='store_true')
    parser.add_argument('--only_supervised', action='store_true')
    parser.add_argument('--train_with_mixcase', action='store_true')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--ckpt_dir',type=str, default="./ckpt")
    parser.add_argument('--log_name', type=str, default='Log')
    parser.add_argument('--mixcase_threshold', type=float, default=0.8)
    parser.add_argument('--transfer_filename', type=str, default=None)
    parser.add_argument('--three_classes', action='store_true')
    parser.add_argument('--finetune', action="store_true")
    parser.add_argument('--mixcase_as_mgt', action="store_true")

    
    args = parser.parse_args()


    # load the DATAFILE
    with open(DATA_FILE, 'r') as file:
        data = json.load(file)
    
    # truncate the training data to args.train_threshold: 10000 and the testing to args.test_threshold: 1000
    if args.train_threshold:
        data['train']['text'] = data['train']['text'][:args.train_threshold]
        data['train']['label'] = data['train']['label'][:args.train_threshold]
    if args.test_threshold:
        data['test']['text'] = data['test']['text'][:args.test_threshold]
        data['test']['label'] = data['test']['label'][:args.test_threshold]

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        print(f"'{args.ckpt_dir}' are created.")
    else:
        print(f"'{args.ckpt_dir}' already exist.")
    DEVICE = args.DEVICE

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')


    base_model_name = args.base_model_name.replace('/', '_')
    SAVE_PATH = f"results/{base_model_name}-{args.mask_filling_model_name}/{args.dataset}"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_PATH)}")

    # write args to file
    with open(os.path.join(SAVE_PATH, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # get generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(
        args.base_model_name, cache_dir)
    load_base_model(base_model, DEVICE)


    def ll_criterion(text): return get_ll(
        text, base_model, base_tokenizer, DEVICE)

    def rank_criterion(text): return -get_rank(text,
                                               base_model, base_tokenizer, DEVICE, log=False)

    def logrank_criterion(text): return -get_rank(text,
                                                  base_model, base_tokenizer, DEVICE, log=True)

    def entropy_criterion(text): return get_entropy(
        text, base_model, base_tokenizer, DEVICE)

    def GLTR_criterion(text): return get_rank_GLTR(
        text, base_model, base_tokenizer, DEVICE)
    
    outputs = []
    
    if args.three_classes:
        # outputs.append(run_threshold_experiment(data, ll_criterion, "likelihood", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
        # outputs.append(run_threshold_experiment(data, rank_criterion, "rank", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
        # outputs.append(run_threshold_experiment(
        #     data, logrank_criterion, "log_rank", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
        # outputs.append(run_threshold_experiment(
        #     data, entropy_criterion, "entropy", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
        # outputs.append(run_GLTR_experiment(data, GLTR_criterion, "rank_GLTR", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
        # outputs.append(run_supervised_experiment(data, model='distilbert-base-uncased',
        #            cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, pos_bit=1, num_labels=3, finetune=True, test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
        outputs.append(run_supervised_experiment(data, model='Hello-SimpleAI/chatgpt-detector-roberta',
                    cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, pos_bit=1, num_labels=3, test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir, finetune=True))
        print("roberta done")
        outputs.append(run_detectgpt_experiments(
            args, data, base_model, base_tokenizer, test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
        print("detectgpt done")
        outputs.append(run_radar(data, DEVICE=DEVICE, finetune=args.finetune, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir, test_only=args.test_only, three_classes=args.three_classes))
        print("radar done")
    else:
        if not args.only_supervised:
            outputs.append(run_threshold_experiment(data, ll_criterion, "likelihood", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
            outputs.append(run_threshold_experiment(data, rank_criterion, "rank", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
            outputs.append(run_threshold_experiment(
                data, logrank_criterion, "log_rank", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
            outputs.append(run_threshold_experiment(
                data, entropy_criterion, "entropy", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
            outputs.append(run_GLTR_experiment(data, GLTR_criterion, "rank_GLTR", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
            # run GPTZero: pleaze specify your gptzero_key in the args
            # outputs.append(run_gptzero_experiment(data, api_key=args.gptzero_key, test_only = args.test_only, no_auc=args.no_auc))
            # run DetectGPT
            outputs.append(run_detectgpt_experiments(
                args, data, base_model, base_tokenizer, test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
        outputs.append(run_sentinel(data, DEVICE=DEVICE, finetune=args.finetune, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir, test_only=args.test_only))
        outputs.append(run_radar(data, DEVICE=DEVICE, finetune=args.finetune, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir, test_only=args.test_only))
        outputs.append(run_supervised_experiment(data, model='roberta-base-openai-detector',
                    cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir, finetune=args.finetune))
        outputs.append(run_supervised_experiment(data, model='Hello-SimpleAI/chatgpt-detector-roberta',
                    cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, pos_bit=1, test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir, finetune=args.finetune))
        outputs.append(run_supervised_experiment(data, model='distilbert-base-uncased',
                    cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, pos_bit=1, finetune=True, test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
        

    # save results 
    with open(f"logs/{args.log_name}", "a") as wf:
        for row in outputs:
            json.dump(row, wf)
            wf.write("\n")

    print("Finish")
