from reg import Critic, Trainer
from reg_data import get_loader
from param import parse_args
import json
import torch
from tqdm import tqdm


def main(dataset='refcoco+', split='testB', task='REG', lr=None):

    args = parse_args()
    args.gpu = 0
    args.train = 'val'
    args.num_beams = 5
    args.batch_size = 16
    args.dataset = dataset
    split_map = {'refcoco+': 'unc',
                 'refcoco': 'unc',
                 'refcocog': 'umd'}
    args.dataset_split = split_map[args.dataset]
    args.ofa_test = True
    if lr != None:
        args.load = '/workspace/yfl/codebase/VL-T5/VL-T5/snap/'+args.dataset+'/' + task + '/' + lr +'/2'
    else:
        args.load = '/workspace/yfl/codebase/VL-T5/VL-T5/snap/' + args.dataset + '/' + task + '/2'
    args.experiment_name = task


    val_loader = get_loader(
        args,
        split=split, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
    )

    trainer = Trainer(args, train=False)
    predictions = trainer.predict(val_loader)

    generate_res = {}  # dict in the form: ref_id: sent

    # This two line code is for my onw generate result.
    for prediction in predictions:
        generate_res[prediction['ref_id']] = prediction['sent']

    assert len(predictions) == len(generate_res), 'The length does not match, prediction is{}, and generate_res is{}'.format(len(predictions), len(generate_res))


    # This is for generate result from my dear duo bro.
    # with open('reg_uniref/reg_uniref_coco.json') as f:
    #     uniref = json.load(f)
    #
    # uniref_result = {}
    # for item in uniref:
    #     uniref_result[item['ref_id']] = item['sent']

    # uniref_result_testA = {}
    # for prediction in predictions:
    #     uniref_result_testA[prediction['ref_id']] = uniref_result[prediction['ref_id']]

    critic = Critic(args)

    score_sum = torch.FloatTensor([0]).cuda()
    score_cnt = torch.FloatTensor([0]).cuda()

    # uniref_score_sum = torch.FloatTensor([0]).cuda()
    # uniref_score_cnt = torch.FloatTensor([0]).cuda()
    #
    # original_score_sum = torch.FloatTensor([0]).cuda()
    # original_score_cnt = torch.FloatTensor([0]).cuda()

    for i, batch in enumerate(tqdm(val_loader, ncols=120, desc="ofa_evaluate")):
        sample_dict = {}
        sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
        sample_dict['refBoxes'] = batch['refBoxes']

        # uniref_sample_dict = {}
        # uniref_sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
        # uniref_sample_dict['refBoxes'] = batch['refBoxes']

        ref_ids = batch['ref_ids']

        sample_sents = []
        uniref_sample_sents = []
        for ref_id in ref_ids:
            sent = generate_res[ref_id]
            sample_sents.append(sent)

            # uniref_sent = uniref_result[ref_id]
            # uniref_sample_sents.append(uniref_sent)

        sample_dict['sents'] = sample_sents  # a list of sent
        # uniref_sample_dict['sents'] = uniref_sample_sents

        # original_sample_dict = {}
        # original_sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
        # original_sample_dict['refBoxes'] = batch['refBoxes']
        # original_sample_dict['sents'] = batch['sents']


        # rewarder should return a tensor in the shape of bacthsize
        scores, _ = critic.compute_score(sample_dict)
        # uniref_results, uniref_scores = critic.compute_score(uniref_sample_dict)
        # original_results, original_scores = critic.compute_score(original_sample_dict)

        score_sum += sum(scores) if scores is not None else 0
        score_cnt += len(scores) if scores is not None else 0

        # uniref_score_sum += sum(uniref_scores) if uniref_scores is not None else 0
        # uniref_score_cnt += len(uniref_scores) if uniref_scores is not None else 0
        #
        # original_score_sum += sum(original_scores) if original_scores is not None else 0
        # original_score_cnt += len(original_scores) if original_scores is not None else 0

    print("Score:{}".format(score_sum.item() / score_cnt.item()))
    print("score_sum:{}".format(score_sum.item()))
    print("score_cnt:{}".format(score_cnt.item()))

    # print("uniref_Score:{}".format(uniref_score_sum.item() / uniref_score_cnt.item()))
    # print("uniref_score_sum:{}".format(uniref_score_sum.item()))
    # print("uniref_score_cnt:{}".format(uniref_score_cnt.item()))
    #
    # print("original_Score:{}".format(original_score_sum.item() / original_score_cnt.item()))
    # print("original_score_sum:{}".format(original_score_sum.item()))
    # print("original_score_cnt:{}".format(original_score_cnt.item()))



if __name__ == '__main__':

    # task_name = 'mmi_scst_exp'
    # main(dataset='refcoco', split='testA', task=task_name)

    # task_name = 'REG'
    # main(dataset='refcoco', split='testA', task=task_name)

    task_name = 'vlt5_ofa_exp_only_sample_with_baseline_no_clamp'
    lr = '1e-06'
    main(dataset='refcoco', split='testA', task=task_name, lr=lr)

