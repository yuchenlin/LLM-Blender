# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch
import transformers
import numpy as np
import wandb
import os
import pprint
import warnings
import logging
from transformers import TrainingArguments
from transformers.trainer_utils import PredictionOutput, is_main_process
warnings.filterwarnings("ignore")
from llm_blender.common.utils import (
    str2bool,
    empty2None,
    seed_everything
)
from llm_blender.pair_ranker.trainer import (
    compute_metrics_for_pairranker,
    compute_metrics_for_scr
)
from llm_blender.pair_ranker.model_util import (
    build_ranker,
    build_tokenizer,
    build_collator,
)
from llm_blender.pair_ranker.data import (
    load_data,
    Dataset
)
from llm_blender.pair_ranker.trainer import (
    RerankerTrainer,
)
from llm_blender.pair_ranker.config import (
    RankerConfig,
)

def main(args):
    seed_everything(args.seed)

    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info(f"device: {device}, n_gpu: {n_gpu}")

    # set up tokenizer
    tokenizer = build_tokenizer(args.model_name, cache_dir=args.cache_dir)
    # set up data collator, also add prefix as new tokens to tokenizer
    data_collator = build_collator(
        args.ranker_type, tokenizer,
        args.source_maxlength, args.candidate_maxlength,
    )

    # set up dataset
    train_dataset = None
    eval_dataset = None
    predict_dataset = None
    if args.do_train:
        train_examples = load_data(args.train_data_path, args, max_size=args.max_train_data_size)
        train_dataset = Dataset(train_examples, args.n_candidates)
    if args.do_eval:
        eval_examples = load_data(args.eval_data_path, args, max_size=args.max_eval_data_size)
        eval_dataset = Dataset(eval_examples, args.n_candidates)
    else:
        args.evaluation_strategy = 'no'
        args.save_strategy = 'no'

    if args.do_predict:
        predict_examples = load_data(args.test_data_path, args, max_size=args.max_predict_data_size)
        predict_dataset = Dataset(predict_examples, args.n_candidates)

    if args.do_train:
        if args.do_eval:
            assert train_dataset.n_tasks == eval_dataset.n_tasks
        args.n_tasks = train_dataset.n_tasks
    elif args.do_predict:
        args.n_tasks = predict_dataset.n_tasks

    # set up model
    
    if args.load_checkpoint:
        config = RankerConfig.from_json_file(os.path.join(args.load_checkpoint, "config.json"))
        for k in args.__dict__:
            if k in config.__dict__:
                print(k, getattr(args, k))
                setattr(config, k, getattr(args, k))
        model = build_ranker(
            args.ranker_type,
            args.model_type,
            args.model_name,
            args.cache_dir,
            config,
            tokenizer,
        )
        state_dict = torch.load(os.path.join(args.load_checkpoint, "pytorch_model.bin"))
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            logging.warning(f"Missing keys: {load_result.missing_keys}")
        else:
            logging.info(f"Successfully loaded checkpoint from '{args.load_checkpoint}'")
    else:
        config = RankerConfig()
        for k, v in args.__dict__.items():
            if k in config.__dict__:
                setattr(config, k, v)
        model = build_ranker(
            args.ranker_type,
            args.model_type,
            args.model_name,
            args.cache_dir,
            config,
            tokenizer,
        )
        logging.info(f"build new model")
    for k, v in args.__dict__.items():
        if k in config.__dict__:
            setattr(config, k, v)

    # set up trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        logging_first_step=args.logging_first_step,
        log_level=args.log_level,
        report_to=args.report_to,
        run_name=args.run_name,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        seed=args.seed,
        local_rank=args.local_rank,
        fp16=args.fp16,
        deepspeed=args.deepspeed, #
        label_names=args.label_names,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        adafactor=args.adafactor,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        disable_tqdm=False,
        greater_is_better=True,
    )

    logging.info(f"training args: {training_args}")
    logging.info(f"model config: {config}")
    if args.ranker_type == "pairranker":
        compute_metrics = compute_metrics_for_pairranker
    else:
        compute_metrics = compute_metrics_for_scr

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.do_train:
        # set up wandb
        if args.report_to == "wandb":
            wandb.init(project="Reranker", group=args.ranker_type, name=args.run_name)
            wandb.config.update(args)
        else:
            os.environ["WANDB_DISABLED"] = 'true'

        if args.evaluate_before_training:
            metrics = trainer.evaluate()
            logging.info(f"Evaluate first step: \n{metrics}")

        logging.info("Start training")
        outputs = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        logging.info("Training finished")
        global_step, training_loss = outputs.global_step, outputs.training_loss
        metrics = outputs.metrics
        logging.info(f"global_step = {global_step}, average loss = {training_loss}")
        for key, value in metrics.items():
            logging.info(f"{key} = {value}")

        if is_main_process(training_args.local_rank):
            logging.info("Saving model")
            best_checkpoint_folder = os.path.join(args.output_dir, "checkpoint-best")
            trainer.save_model(best_checkpoint_folder)

    if args.do_predict:
        logging.info("Start predicting")
        outputs: PredictionOutput = trainer.predict(predict_dataset)
        predictions = outputs.predictions
        labels = outputs.label_ids
        metrics = outputs.metrics
        logging.info(f"metrics: {metrics}")

        # save predictions
        if args.save_predictions and is_main_process(training_args.local_rank):
            if args.output_dir is None:
                raise ValueError("output_dir must be set to save predictions")
            output_path = os.path.join(args.output_dir, "predictions.pt")
            if args.ranker_type == "pairranker" and args.inference_mode == "full":
                # predictions[0] is a [size, num_candidate, num_candidates] ndarray, which stores the comparison results of each candidate with all other candidates
                output_path = os.path.join(args.output_dir, "predictions_full.pt")
            elif args.ranker_type == "pairranker" and args.inference_mode == "bubble":
                output_path = os.path.join(args.output_dir, "predictions_bubble.pt")
            else:
                output_path = os.path.join(args.output_dir, "predictions.pt")

            with open(output_path, "wb") as f:
                torch.save(predictions, f)
            with open(os.path.join(args.output_dir, "labels.pt"), "wb") as f:
                torch.save(labels, f)
            logging.info(f"predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--ranker_type", type=str, choices=[
        "summareranker", "dual", "pairranker"
    ], default="sc")
    parser.add_argument("--model_type", type=str, choices=[
        "roberta", "bert", "t5", 'deberta', 'xlm-roberta', 'flan-t5', 'alpaca', 'opt', 'phi'
    ], default="roberta")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--load_checkpoint", type=empty2None, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--loss_type", type=str, choices=[
      "BCE", "MSE", "instructgpt", "MoE_BCE", "simcls", "open_instruct_BCE"
    ], default="BCE")

    # data config
    parser.add_argument("--n_candidates", type=int, default=-1)
    parser.add_argument("--candidate_decoding_method", type=empty2None, default=None, help="separted by comma. Empty string for all methods")
    parser.add_argument("--candidate_model", type=empty2None, default=None, help="separted by comma. Empty string for all models")
    parser.add_argument("--source_maxlength", type=int, default=256)
    parser.add_argument("--candidate_maxlength", type=int, default=256)
    parser.add_argument("--num_pos", type=int, default=1)
    parser.add_argument("--num_neg", type=int, default=1)
    parser.add_argument("--sub_sampling_ratio", type=float, default=0.4)
    parser.add_argument("--sub_sampling_mode", type=str, choices=[
        "uniform", "top_bottom", "top_random", "random_bottom", "random", 
        "uniform", "all_pair"
    ], default="top_bottom")
    parser.add_argument("--max_train_data_size", type=int, default=-1)
    parser.add_argument("--max_eval_data_size", type=int, default=-1)
    parser.add_argument("--max_predict_data_size", type=int, default=-1)
    parser.add_argument("--using_metrics", type=str, default="rouge1,rouge2,rougeLsum", help="Metrics used for training")

    # running config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--fp16', type=str2bool, default=True)
    parser.add_argument('--deepspeed', type=str, default=None) # "ds_config.json"
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank. Necessary for using the torch.distributed.launch utility.")

    # mode
    parser.add_argument("--do_train", type=str2bool, default=True)
    parser.add_argument("--do_eval", type=str2bool, default=True)
    parser.add_argument("--do_predict", type=str2bool, default=True)

    # training hyperparameters
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=0) # Overrides any effect of :obj:`warmup_ratio`.
    parser.add_argument("--lr_scheduler_type", type=str, choices=[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ], default="linear")
    parser.add_argument('--adafactor', type=bool, default=True)

    # evaluation hyperparameters
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--evaluate_before_training", type=str2bool, default=False)
    parser.add_argument("--evaluation_strategy", type=str, choices=[
        "steps", "epoch", "no"
    ], default="epoch")
    parser.add_argument("--eval_steps", type=int, default=0)

    # predict config
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--save_predictions", type=str2bool, default=True)

    # logging
    parser.add_argument("--logging_first_step", type=str2bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--log_level", type=str, default="passive",
        choices=["passive", "info", "debug", "warning", "error", "critical"])
    parser.add_argument("--report_to", type=str, default='none')
    parser.add_argument("--run_name", type=str, default="basic") # wandb run name

    # save config
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", type=str2bool, default=False)
    parser.add_argument("--save_strategy", type=str, choices=[
        "steps", "epoch", "no"
    ], default="epoch")
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=4)

    # metrics config
    parser.add_argument("--load_best_model_at_end", type=str2bool, default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--metric_for_best_model", type=str, default="dev_score")

    # inference config
    parser.add_argument("--inference_mode", type=str, default="bubble",
        choices=["bubble", "full"])

    # init args
    args = parser.parse_args()
    args.load_best_model_at_end = args.do_train and args.do_predict
    # set up default output dir
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.ranker_type}/{args.model_name}/{args.run_name}"
    args.cache_dir = "./hf_models/" + args.model_name.split('/')[-1] + "/"
    args.label_names = ["scores"]
    args.candidate_decoding_methods = args.candidate_decoding_method.split(',') if args.candidate_decoding_method is not None else None
    args.candidate_models = args.candidate_model.split(',') if args.candidate_model is not None else None
    args.local_rank = os.environ.get("LOCAL_RANK", args.local_rank)
    args.metrics = args.using_metrics.split(',')

    # set up logging
    if args.log_level == "passive":
        args.log_level = "info"
    logging.basicConfig(level="INFO")
    logging.info("args: %s", args)
    main(args)
