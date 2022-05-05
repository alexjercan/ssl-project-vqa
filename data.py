import os
import wget
import zipfile

data_path = "../data"
root_path = os.path.join(data_path, "okvqa")

train_images_url = "http://images.cocodataset.org/zips/train2014.zip"
test_images_url = "http://images.cocodataset.org/zips/val2014.zip"

train_question_url = "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip"
test_questions_url = (
    "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip"
)

train_annotations_url = (
    "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip"
)
test_annotations_url = (
    "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip"
)

urls = [
    train_images_url,
    test_images_url,
    train_question_url,
    test_questions_url,
    train_annotations_url,
    test_annotations_url,
]

zip_paths = list(map(lambda url: os.path.join(data_path, url.split("/")[-1]), urls))


def download_zip(url, path, force: bool = False):
    if os.path.exists(path):
        print(f"{path} found. skipping...")
        return

    wget.download(url, out=path)


def download_okvqa(force: bool = False) -> None:
    if os.path.exists(root_path) and not force:
        print("Dataset root dir found. skiping...")
        return

    os.makedirs(root_path, exist_ok=True)

    for url, path in zip(urls, zip_paths):
        download_zip(url, path, force)

    for path in zip_paths:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(root_path)


if __name__ == "__main__":
    download_okvqa()

