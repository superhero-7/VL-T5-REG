from param import parse_args
from reg import Trainer
from reg_data import get_loader
import json

def test(dataset='refcoco+', split='testB', task='REG', save=False):

    args = parse_args()
    args.gpu = 0
    args.train = 'val'
    args.num_beams = 5
    args.batch_size = 100
    args.dataset = dataset
    split_map = {'refcoco+': 'unc',
                 'refcoco': 'unc',
                 'refcocog': 'umd'}
    args.dataset_split = split_map[args.dataset]
    args.load = '/workspace/yfl/codebase/VL-T5/VL-T5/snap/'+args.dataset+'/' + task + '/BEST'
    args.experiment_name = task


    val_loader = get_loader(
        args,
        split=split, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
    )


    trainer = Trainer(args, train=False)

    # data = json.dumps(results)
    # with open('refcoco+_testB', 'w') as f:
    #     f.write(data)
    #
    # print(results)

    Score, results = trainer.evaluate(val_loader, save=True)

    # print(len(Score['CIDErs']))
    # if save:
    #     i = 0
    #     for item in results:
    #         item['cider'] = Score['CIDErs'][i]
    #         item['meteor'] = Score['METEORs'][i]
    #         i = i+1
    #
    #     data = json.dumps(results)
    #     if mmi:
    #         with open('result/'+args.dataset+'_'+split+'_mmi.json', 'w') as f:
    #             f.write(data)
    #     else:
    #         with open('result/'+args.dataset+'_'+split+'.json', 'w') as f:
    #             f.write(data)


if __name__ == '__main__':


    # test(dataset='refcoco', split='testA', task='REG', save=False)
    # test(dataset='refcoco', split='testB', task='REG', save=False)
    # test(dataset='refcoco', split='val', task='REG', save=False)

    # test(dataset='refcoco+', split='testA', task='REG', save=False)

    # test(dataset='refcoco+', split='testB', task='REG', save=False)
    # test(dataset='refcoco+', split='testB', task='REG_mmi', save=False)
    # test(dataset='refcoco+', split='testB', task='scst_exp_vlt5_ofa', save=False)
    # test(dataset='refcoco+', split='testB', task='mmi_scst_exp', save=False)
    # test(dataset='refcoco+', split='testB', task='scst_exp_lr', save=False)
    test(dataset='refcoco+', split='testB', task='REG_scst', save=False)
    # test(dataset='refcoco+', split='val', task='REG', save=False)


