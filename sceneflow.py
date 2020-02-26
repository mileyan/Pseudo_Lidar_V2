import os
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--force', action='store_true')
args = parser.parse_args()


def main():
    assert os.path.isdir(args.path), "Path is not correct"
    path = args.path
    monkaa = path + "/monkaa/frames_cleanpass"
    monkaa_disparity = path + "/monkaa/disparity"
    driving = path + "/driving/frames_cleanpass"
    driving_disparity = path + "/driving/disparity"
    flyingthings3d = path+"/flyingthings3d/frames_cleanpass"
    flyingthings3d_disparity = path+"/flyingthings3d/disparity"

    if args.force and os.path.isdir('sceneflow'):
        shutil.rmtree('sceneflow')
    if os.path.isdir('sceneflow'):
        print('Soft links are existing. Stop running')
        return
    os.makedirs('sceneflow')
    os.system('ln -s {} sceneflow/monkaa_cleanpass'.format(monkaa))
    os.system('ln -s {} sceneflow/monkaa_disparity'.format(monkaa_disparity))
    os.system('ln -s {} sceneflow/flyingthings3d_cleanpass'.format(flyingthings3d))
    os.system(
        'ln -s {} sceneflow/flyingthings3d_disparity'.format(flyingthings3d_disparity))
    os.system('ln -s {} sceneflow/driving_cleanpass'.format(driving))
    os.system('ln -s {} sceneflow/driving_disparity'.format(driving_disparity))


if __name__ == '__main__':
    main()