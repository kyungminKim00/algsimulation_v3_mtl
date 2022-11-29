import os
import shutil

from header.index_forecasting import RUNHEADER

search_dir = "./save/result/"
for model_name in os.listdir(search_dir):
    if len(model_name.split("_")) > 8:  # is model directoties
        for mode in ["", "validation"]:
            model_dir = f"{search_dir}{model_name}/{mode}"

            for market_name in os.listdir(model_dir):
                if market_name in list(
                    RUNHEADER.mkname_mkidx.keys()
                ):  # is market directoties
                    result_index = f"{model_dir}/index"
                    result_return = f"{model_dir}/return"

                    for fn in os.listdir(result_index):
                        if "jpeg" in fn:
                            target = f"./save/result/{model_name}/index/"
                            if os.path.isdir(target):
                                os.makedirs(target)
                                shutil.copy2(
                                    f"{result_index}/{fn}",
                                    f"{target}{market_name}_{fn}",
                                )

                    for fn in os.listdir(result_return):
                        if "jpeg" in fn:
                            target = f"./save/result/{model_name}/return/"
                            if os.path.isdir(target):
                                os.makedirs(target)
                                shutil.copy2(
                                    f"{result_return}/{fn}",
                                    f"{target}{market_name}_{fn}",
                                )
