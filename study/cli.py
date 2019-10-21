import click
import os
from . import make_dataset, image, video, const, anfis, logger
from .util.path import base_path, dataset_path, video_path

@click.group("cli")
@click.option("--log", default="INFO", help="Logging level of the logger.")
def cli(log):
    logger.set_level(log)


@cli.command("image")
def img():
    image.main()


@cli.command("video")
@click.option("--path")
@click.option('--save', default='False')
def vid(path, save):
    const.set('VIDEO_PATH', video_path(path + '.' + const.get('EXT_VIDEO')))
    video.main(path=const.get('VIDEO_PATH'), save=save)


@cli.command("make_dataset")
@click.option('--video')
@click.option('--feature')
def dataset(video, feature):
    const.set('FEATURE', feature)
    make_dataset.main(video, feature)


@cli.command("anfis")
@click.option('--feature', default='glcm')
@click.option('--video', default='1')
@click.option('--epochs', default=const.get('EPOCHS'))
@click.option('--verbose', is_flag=True)
def anf(feature, video, epochs, verbose):
    const.set('FEATURE', feature.split(','))

    dataset_paths = {
        'car'   : [],
        'truck' : [],
    }

    video_name = const.get('VIDEO_NAME')[int(video)]

    for feature in const.get('FEATURE'):
        car_path = dataset_path(os.path.join(video_name, 'dataset_{0}_for_c.csv'.format(feature)))
        truck_path = dataset_path(os.path.join(video_name, 'dataset_{0}_for_t.csv'.format(feature)))
        
        dataset_paths['car'].append(car_path)
        dataset_paths['truck'].append(truck_path)

        if verbose:
            click.echo('feature ' + feature)
            click.echo('use {0}'.format(car_path))
            click.echo('use {0}'.format(truck_path))
            click.echo()

    const.set('VIDEO_PATH', video_path(video_name) + '.' + const.get('EXT_VIDEO'))

    const.set('EPOCHS', epochs)

    if verbose:
        const.set('VERBOSE', True)

    anfis.main(dataset_paths)
