import os
import shutil
import sys
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fileListTxt', help='text file', type=str)
    parser.add_argument('fileDir', help='images directory', type=str)
    parser.add_argument('fileType', help='image type', type=str)


    args = parser.parse_args()
    return args



def pick(fileListTxt, fileDir, fileType):
    pickedPath = f"/content/{fileType}"

    fileDir = os.path.abspath(fileDir)
    if fileDir[-1] == "/":
        fileDir = fileDir[:-1]

    fileList = os.listdir(fileDir)

    # recreate folder
    if os.path.exists(pickedPath):
        shutil.rmtree(pickedPath)
    os.makedirs(pickedPath)

    file_ = open(fileListTxt, 'r')
    goodfileList = [f.strip() for f in file_.readlines()]
    file_.close()

    for f in tqdm(fileList):
        f = f.strip()
        fileName = os.path.basename(f)
        txtName = fileName.split('.')[0] + '.txt'
        if fileName not in goodfileList:
            continue

        srcFile = os.path.join(fileDir, f)
        dstFile = os.path.join(pickedPath, fileName)
        txtsrcFile = os.path.join('/content/augmented/txt/', txtName)
        txtdstFile = os.path.join(pickedPath, txtName)

        shutil.copyfile(srcFile, dstFile)
        shutil.copyfile(txtsrcFile, txtdstFile)
    print("Path of picked images = ",os.path.abspath(pickedPath))

if __name__ == '__main__':
    args = parse_args()
    fileListTxt = args.fileListTxt
    fileDir = args.fileDir
    fileType = args.fileType

    if not os.path.exists(fileListTxt):
        print("Error !!! %s is not exists, please check the parameter"%fileListTxt)
        sys.exit(0)

    if not os.path.exists(fileDir):
        print("Error !!! %s is not exists, please check the parameter"%fileDir)
        sys.exit(0)

    pick(fileListTxt, fileDir, fileType)
    print("Done!")
