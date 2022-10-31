import re
import os
from PIL import Image
import glob
import platform
import header.index_forecasting.RUNHEADER as RUNHEADER


def animate_gif(source_dir, duration=200):
    width = int(RUNHEADER.img_jpeg["width"]) * 0.5
    height = int(RUNHEADER.img_jpeg["height"]) * 0.5

    for dir_name in os.listdir(source_dir):
        for target, mode in [
            ["/fig_index/return", "test"],
            ["/validation/fig_index/return", "validation"],
        ]:
            img, *imgs = None, None
            t_dir = source_dir + "/" + dir_name + target

            # filepaths
            fp_in = t_dir + "/*.jpeg"
            fp_out = t_dir + "/" + dir_name + "_" + mode + ".gif"

            try:
                print("Reading:{}".format(fp_in))
                img, *imgs = [
                    Image.open(f).resize((int(width), int(height)))
                    for f in sorted(glob.glob(fp_in))
                ]
                img.save(
                    fp=fp_out,
                    format="GIF",
                    append_images=imgs,
                    save_all=True,
                    duration=duration,
                )
                img.close()
            except ValueError:
                print("ValueError:{}".format(fp_in))
                pass


if __name__ == "__main__":
    animate_gif("./save/result", duration=200)
