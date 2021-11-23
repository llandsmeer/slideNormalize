#!/usr/bin/env python3

import os
import sys
import argparse
import cv2
import time

__doc__ = '''
slideNormalize: normalizes an image by assessing image
texture using OpenCV's CLAHE (Contrast Limited Adaptive
Histogram Equalization)'''.strip().replace('\n', ' ')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
        '-f', '--file', metavar='FILE', required=True,
        help="the filename of the NDPI/TIF file to process.")
    parser.add_argument(
        '-e', '--ext', metavar='EXT',
        help="the standard output-filename extension of the image-tile to process.")
    parser.add_argument(
        '-o', '--out', metavar='FILE',
        help="the output-filename of the NDPI/TIF file to process [default: replace insert .normalized before .tile.tissue.png]")
    parser.add_argument(
        '-s', '--show', default=False, action='store_true',
        help="show results in a graphical interface")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = args.file.replace('.tile.tissue.png', '.normalized.tile.tissue.png')
    if args.ext is not None:
        if '.' in args.out:
            args.out = args.out[:args.out.rfind('.')]
        args.out = args.out + '.' + args.ext
    return args

def test_parse_args():
    args = parse_args(['-f', 'input', '-e', 'ext'])
    assert args.out == 'input.ext'
    args = parse_args(['-f', 'input.png', '-e', 'ext'])
    assert args.out == 'input.ext'
    args = parse_args(['-f', 'x0.y0.tile.tissue.png'])
    assert args.out == 'x0.y0.normalized.tile.tissue.png'

def normalize(infile, outfile, show=False):
    bgr = cv2.imread(infile)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(outfile, bgr_out)
    if show:
        preview = (512, 512)
        cv2.imshow('original (preview)', cv2.resize(bgr, preview))
        cv2.imshow('normalized (preview)', cv2.resize(bgr_out, preview))
        print('  preview: press any key to exit..')
        cv2.waitKey()

def main():
    args = parse_args()
    if not os.path.exists(args.file):
        print('Could not find input file', repr(args.file), file=sys.stderr)
        exit(1)
    if args.file == args.out:
        print('Refusing to overwrite input file with output file', repr(args.file), file=sys.stderr)
        exit(1)
    print('Running slideNormalize')
    print('  input    ', args.file, file=sys.stderr)
    print('  outfile  ', args.out, file=sys.stderr)
    tic = time.time()
    normalize(args.file, args.out, args.show)
    toc = time.time()
    print(f'  done (took {toc-tic:.2f}s)')

if __name__ == '__main__':
    main()
