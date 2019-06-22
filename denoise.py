import sys
import os


inputFile = sys.argv[1]
outputFile = sys.argv[2]

# for i in range(164):

# inputFile = './well_voice/toWAV/1' + '{:05d}'.format(i+1) + '.wav'
# print(inputFile)
# outputFile = './denoise/well_denoise/1' + '{:05d}'.format(i+1) + '.wav'
# print(outputFile)

# transform input file to .pcm
print('transforming input file to pcm')
cmd = 'ffmpeg -y -i ' + inputFile + ' -acodec pcm_s16le -f s16le -ac 1 -ar 48000 input.pcm'
os.system(cmd)

# denoise
print('applying denoise operation')
cmd = './rnnoise_demo input.pcm output.pcm'
os.system(cmd)

# transform denoised pcm file to output file
print('producing output file')
cmd = 'ffmpeg -f s16le -ar 24k -ac 2 -i output.pcm ' + outputFile
os.system(cmd)

# remove meta data
cmd = 'rm input.pcm output.pcm'
os.system(cmd)
