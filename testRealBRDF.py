import utils
import torch
import numpy as np
import torch.optim as optim
import argparse
import random
import os
import models
import modelLight
import renderWindow
import renderVisLamp
import renderInvLamp
import torchvision.utils as vutils
import torchvision.models as vmodels
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path as osp
# import lossFunctions
import scipy.ndimage as ndimage
import renderShadowDepth
import pickle
import glob
import cv2


parser = argparse.ArgumentParser()
# The directory of trained models
parser.add_argument('--experimentBRDF', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentVisLamp', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentInvLamp', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentVisWindow', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentInvWindow', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentDirecIndirec', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentShadow', default=None, help='the path to store samples and models')
parser.add_argument('--testList', default=None, help='the path to store samples and models' )

# The basic training setting
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network' )
parser.add_argument('--envRow', type=int, default=120, help='the height / width of the input image to network' )
parser.add_argument('--envCol', type=int, default=160, help='the height / width of the input image to network' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network' )

# Training epochs and iterations
parser.add_argument('--nepochBRDF', type=int, default=15, help='the epoch used for testing' )
parser.add_argument('--iterIdVisLamp', type=int, default=119540, help='the iteration used for testing' )
parser.add_argument('--iterIdInvLamp', type=int, default=150000, help='the iteration used for testing' )
parser.add_argument('--iterIdVisWin', type=int, default=120000, help='the iteration used for testing' )
parser.add_argument('--iterIdInvWin', type=int, default=200000, help='the iteration used for testing' )
parser.add_argument('--iterIdDirecIndirec', type=int, default=180000, help='the iteration used for testing' )
parser.add_argument('--iterIdShadow', type=int, default=70000, help='the iteration used for testing')

parser.add_argument('--isOptimize', action='store_true', help='use optimization for light sources or not' )
parser.add_argument('--iterNum', type=int, default = 400, help='the number of interations for optimization')

# The training weight
parser.add_argument('--shadingWeight', type=float, default=1.0, help='the weight for shading error' )
parser.add_argument('--geometryWeight', type=float, default=1.0, help='the weight for geometry error' )
parser.add_argument('--sizeWeight', type=float, default=0.2, help='the weight for size error' )
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for render error' )
parser.add_argument('--winSrcIntWeight', type=float, default=0.001, help='the loss for window light source' )
parser.add_argument('--winSrcAxisWeight', type=float, default=1.0, help='the loss for window light source' )
parser.add_argument('--winSrcLambWeight', type=float, default=0.001, help='the loss for window light source' )

# Starting and ending point
parser.add_argument('--rs', type=int, default=0, help='starting point' )
parser.add_argument('--re', type=int, default=1, help='ending point' )

parser.add_argument('--fovX', type=float, default=57.95)


# The detail network setting
opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True )

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )

if opt.experimentBRDF is None:
    opt.experimentBRDF = 'check_brdf_w%d_h%d' % (opt.imWidth, opt.imHeight )
opt.experimentBRDF = osp.join(curDir, opt.experimentBRDF )

# Network for BRDF prediction
encoder = models.encoder(isGtGeometry = True )
albedoDecoder = models.decoder(mode=0 )
normalDecoder = models.decoder(mode=1 )
roughDecoder = models.decoder(mode=2 )

# Network for lighting prediction
encoder.load_state_dict(torch.load('{0}/encoder_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
for param in encoder.parameters():
    param.requires_grad = False
albedoDecoder.load_state_dict(torch.load('{0}/albedo_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
for param in albedoDecoder.parameters():
    param.requires_grad = False
normalDecoder.load_state_dict(torch.load('{0}/normal_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
for param in normalDecoder.parameters():
    param.requires_grad = False
roughDecoder.load_state_dict(torch.load('{0}/rough_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
for param in roughDecoder.parameters():
    param.requires_grad = False

# Send things into GPU
encoder = encoder.cuda()
albedoDecoder = albedoDecoder.cuda()
roughDecoder = roughDecoder.cuda()
normalDecoder = normalDecoder.cuda()

with open(opt.testList, 'r') as fIn:
    dirList = fIn.readlines()
dirList = [x.strip() for x in dirList if x[0] != '#' ]

timestart = torch.cuda.Event(enable_timing = True )
timestop = torch.cuda.Event(enable_timing = True )


for dataId in range(max(opt.rs, 0), min(opt.re, len(dirList ) ) ):
    dataDir = dirList[dataId ]
    print(dataDir )
    # Load image, assume the the longest width will be 320/1600
    inputDir = osp.join(dataDir, 'input')
    outputDir = osp.join(dataDir, 'BRDF')

    if not osp.isdir(outputDir ):
        os.system('mkdir %s' % outputDir )

    imName = osp.join(inputDir, 'im.png')
    depthName = osp.join(inputDir, 'depth.npy')

    im = cv2.imread(imName )[:, :, ::-1 ]
    originHeight, originWidth = im.shape[0:2 ]
    width = opt.imWidth
    height = int(float(originWidth) / float(width) * originHeight )
    if width != originWidth:
        im = cv2.resize(im, (width, height ), interpolation = cv2.INTER_AREA )
    sWidth, sHeight = int(width / 2.0), int(height / 2.0 )

    # depth size should be height x width
    depth = np.load(depthName )

    if width != originWidth:
        depth = cv2.resize(depth, (width, height), interpolation = cv2.INTER_AREA )

    imBatch = im.transpose(2, 0, 1)[np.newaxis, :, :].astype(np.float32 ) / 255.0
    imBatch = torch.from_numpy(imBatch ** (2.2 )  ).cuda()

    depthBatch = depth[np.newaxis, np.newaxis, :, :].astype(np.float32 )
    depthBatch = torch.from_numpy(depthBatch ).cuda()

    inputBatch = torch.cat([imBatch, depthBatch ], dim=1 )

    # Predict the large BRDF
    timestart.record()
    x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
    albedoPred, af = albedoDecoder(x1, x2, x3,
            x4, x5, x6, [height, width ] )
    normalPred, nf = normalDecoder(x1, x2, x3,
            x4, x5, x6, [height, width] )
    roughPred, rf = roughDecoder(x1, x2, x3,
            x4, x5, x6, [height, width] )
    depthPred = depthBatch

    timestop.record()
    torch.cuda.synchronize()
    print('BRDF time: %.3f ms' % timestart.elapsed_time(timestop ) )

    # Save the BRDF predictions
    albedoName = osp.join(outputDir, 'albedo.npy' )
    albedoImName = osp.join(outputDir, 'albedo.png' )
    albedoPred = albedoPred.detach().cpu().numpy()
    albedoPredIm = albedoPred.squeeze().transpose(1, 2, 0)
    albedoPredIm = ( (albedoPredIm **(1.0/2.2) ) * 255).astype(np.uint8 )
    np.save(albedoName, albedoPred )
    cv2.imwrite(albedoImName, albedoPredIm[:, :, ::-1] )

    roughName = osp.join(outputDir, 'rough.npy')
    roughImName = osp.join(outputDir, 'rough.png')
    roughPred = roughPred.detach().cpu().numpy()
    roughPredIm = roughPred.squeeze()
    roughPredIm = ( (roughPredIm+1) * 0.5 * 255).astype(np.uint8 )
    np.save(roughName, roughPred )
    cv2.imwrite(roughImName, roughPredIm )

    normalName = osp.join(outputDir, 'normal.npy')
    normalImName = osp.join(outputDir, 'normal.png')
    normalPred = normalPred.detach().cpu().numpy()
    normalPredIm = normalPred.squeeze().transpose(1, 2, 0 )
    normalPredIm = ( 0.5*(normalPredIm + 1)*255 ).astype(np.uint8 )
    np.save(normalName, normalPred )
    cv2.imwrite(normalImName, normalPredIm[:, :, ::-1] )

    depthName = osp.join(outputDir, 'depth.npy')
    depthImName = osp.join(outputDir, 'depth.png')
    depthPred = depthPred.detach().cpu().numpy()
    depthPredIm = depthPred.squeeze()
    depthPredIm = ( 1 / (depthPredIm + 1)  * 255).astype(np.uint8 )
    np.save(depthName, depthPred )
    cv2.imwrite(depthImName, depthPredIm )

    # Save the images
    imName = osp.join(outputDir, 'im.npy' )
    imPngName = osp.join(outputDir, 'im.png' )
    imBatch = imBatch.detach().cpu().numpy()
    imBatchIm = imBatch.squeeze().transpose(1, 2, 0)
    imBatchIm = ( (imBatchIm **(1.0/2.2) ) * 255).astype(np.uint8 )
    np.save(imName, imBatch )
    cv2.imwrite(imPngName, imBatchIm[:, :, ::-1] )

