import click
import os
from . import make_dataset, image, video, const, anfis, logger
from .util.path import base_path, dataset_path

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
    const.set('VIDEO_PATH', video_path(video + '.' + const.get('EXT_VIDEO')))
    video.main(path=const.get('VIDEO_PATH'), save=save)


@cli.command("make_dataset")
@click.option('--video')
@click.option('--feature')
def dataset(video, feature):
    const.set('FEATURE', feature)
    make_dataset.main(video, feature)


@cli.command("anfis")
@click.option('--feature', default='glcm')
@click.option('--video', default='')
@click.option('--epochs', default=const.get('EPOCHS'))
def anf(feature, video, epochs):
    const.set('FEATURE', feature)

    dataset_paths = {
        'car'   : dataset_path(os.path.join(video, 'dataset_{0}_for_c.csv'.format(feature))),
        'truck' : dataset_path(os.path.join(video, 'dataset_{0}_for_t.csv'.format(feature)))
    }

    click.echo('use {0}'.format(dataset_paths['car']))
    click.echo('use {0}'.format(dataset_paths['truck']))
    click.echo()

    const.set('VIDEO_PATH', video_path(video) + '.' + const.get('EXT_VIDEO'))

    anfis.main(dataset_paths, epochs)
