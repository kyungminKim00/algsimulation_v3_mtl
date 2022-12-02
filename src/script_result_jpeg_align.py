import os
import shutil

from header.index_forecasting import RUNHEADER


def extract(model_name):
    if len(model_name.split("_")) > 8:  # is model directoties
        for mode in ["", "validation"]:
            model_dir = (
                f"{search_dir}{model_name}"
                if mode == ""
                else f"{search_dir}{model_name}/{mode}"
            )

            for market_name in os.listdir(model_dir):
                if market_name in list(
                    RUNHEADER.mkname_mkidx.keys()
                ):  # is market directoties
                    for submodel in os.listdir(f"{model_dir}/{market_name}"):
                        if ("txt" not in submodel) and ("csv" not in submodel):
                            for result in [
                                f"{model_dir}/{market_name}/{submodel}/index",
                                f"{model_dir}/{market_name}/{submodel}/return",
                            ]:
                                for fn in os.listdir(result):
                                    target = (
                                        f"./save/result/aligned/{model_name}/index/"
                                        if "/index" in result
                                        else f"./save/result/aligned/{model_name}/return/"
                                    )
                                    if "jpeg" in fn:

                                        if not os.path.isdir(target):
                                            os.makedirs(target)

                                        shutil.copy2(
                                            f"{result}/{fn}",
                                            f"{target}{market_name}_{mode}_{fn}",
                                        )


search_dir = "./save/result/"
model_name = "IF_TOTAL_T20_20221130_1159_m7_1_v26_20221130_1225_2616/"
if model_name == "":
    for model_name in os.listdir(search_dir):
        extract(model_name)
else:
    extract(model_name)
