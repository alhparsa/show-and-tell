from cocoeval.pycocoevalcap.eval import COCOEvalCap
from cocoeval.pycocotools.coco import COCO
from json import encoder
import pylab
import json

def main(args):
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    dataDir='.'
    dataType='val2014'
    annFile='%s/cocoeval/annotations/captions_%s.json'%(dataDir,dataType)
    subtypes=['results', 'evalImgs', 'eval']
    f = args.file_path
    coco = COCO(annFile)
    cocoRes = coco.loadRes(f)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str,
                    default='./cocoeval/results/showattendtell.json', help='Path to generated json file')      
    args = parser.parse_args()
    main(args)
