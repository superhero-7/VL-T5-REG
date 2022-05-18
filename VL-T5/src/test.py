# from tokenization import VLT5TokenizerFast
#
# tokenizer = VLT5TokenizerFast.from_pretrained('t5-base')
#
# ids = tokenizer.convert_tokens_to_ids('<vis_extra_id_0>')
# vocab = tokenizer.get_vocab()
#
# tokens = tokenizer.tokenize('I am super handsome!')
# print(tokens)
#
# print(tokenizer.pad_token_id)
#
#
# for k, v in vocab.items():
#     if v == 1:
#         print(k)
#         break


# import h5py
# #
# # # f = h5py.File('/workspace/yfl/codebase/VL-T5/datasets/RefCOCO/refcocog/features/train_boxes_mattnet.h5', 'r')
# # f = h5py.File('/workspace/yfl/datasets/ref/refcocog/features/val_obj36.h5', 'r')
# # #f = h5py.File('/workspace/yfl/codebase/VL-T5/datasets/COCO/features/train2014_GT.h5', 'r')
# f = h5py.File('/workspace/yfl/datasets/ref/refcocog/features/val_target.h5', 'r')
# i = 0
# for x in f.keys():
#     for item in f[x].keys():
#         print(f[x][item].name)
#         print(f[x][item][()])
#         print("==========================")
#     print("Length of features:{}".format(len(f[x]['features'])))
#     print("Length of objectors:{}".format(f[x]['num_objects'][()]))
#     i = i+1
#     print("+++++++++++++++++++++++++++++++++++++++++++")
#     if i == 3:
#         break
# # print(f['/100022/obj_id'][()])
# # #
# # img_w = f['100022']['img_w']
# # img_g = f['100022']['img_w']
# #
# # print(img_g, img_w)

from param import parse_args
from reg import Trainer
from reg_data import  get_loader
import json

def test(dataset='refcoco+', split='testB', mmi=None, save=False):
    mmi = mmi
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
    split = split
    if mmi:
        args.load = '/workspace/yfl/codebase/VL-T5/VL-T5/snap/'+args.dataset+'/REG_mmi/BEST'
    else:
        args.load = '/workspace/yfl/codebase/VL-T5/VL-T5/snap/'+args.dataset+'/REG/BEST'


    val_loader = get_loader(
        args,
        split=split, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
    )


    trainer = Trainer(args, train=False)

    results = trainer.predict(val_loader)
    # data = json.dumps(results)
    # with open('refcoco+_testB', 'w') as f:
    #     f.write(data)
    #
    # print(results)

    Score = trainer.evaluate(val_loader)

    # print(len(Score['CIDErs']))
    if save:
        i = 0
        for item in results:
            item['cider'] = Score['CIDErs'][i]
            item['meteor'] = Score['METEORs'][i]
            i = i+1

        data = json.dumps(results)
        if mmi:
            with open('result/'+args.dataset+'_'+split+'_mmi.json', 'w') as f:
                f.write(data)
        else:
            with open('result/'+args.dataset+'_'+split+'.json', 'w') as f:
                f.write(data)

if __name__ == '__main__':

    # for mmi in [True,False]:
    #     for split in ['testA', 'testB']:
    #         test(mmi, split=split)

    test(dataset='refcocog', split='val', mmi=True, save=False)
