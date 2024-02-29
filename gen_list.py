import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filelist', type=str, default='finalfilelist.txt')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--cats', type=list, default=[
                    "airplane", "car", "chair", "display", "loudspeaker", "table", "telephone"])
parser.add_argument('--count', type=int, default=1250)
parser.add_argument('--train', type=float, default=0.8)
parser.add_argument('--test', type=float, default=0.1)
parser.add_argument('--val', type=float, default=0.1)
parser.add_argument('--dir', type=str, default='.')
parser.add_argument('--output', type=str, default='final.json')

category_mapping = {
    "04379243": "table",
    "03593526": "jar",
    "04225987": "skateboard",
    "02958343": "car",
    "02876657": "bottle",
    "04460130": "tower",
    "03001627": "chair",
    "02871439": "bookshelf",
    "02942699": "camera",
    "02691156": "airplane",
    "03642806": "laptop",
    "02801938": "basket",
    "04256520": "sofa",
    "03624134": "knife",
    "02946921": "can",
    "04090263": "rifle",
    "04468005": "train",
    "03938244": "pillow",
    "03636649": "lamp",
    "02747177": "trash bin",
    "03710193": "mailbox",
    "04530566": "watercraft",
    "03790512": "motorbike",
    "03207941": "dishwasher",
    "02828884": "bench",
    "03948459": "pistol",
    "04099429": "rocket",
    "03691459": "loudspeaker",
    "03337140": "file cabinet",
    "02773838": "bag",
    "02933112": "cabinet",
    "02818832": "bed",
    "02843684": "birdhouse",
    "03211117": "display",
    "03928116": "piano",
    "03261776": "earphone",
    "04401088": "telephone",
    "04330267": "stove",
    "03759954": "microphone",
    "02924116": "bus",
    "03797390": "mug",
    "04074963": "remote",
    "02808440": "bathtub",
    "02880940": "bowl",
    "03085013": "keyboard",
    "03467517": "guitar",
    "04554684": "washer",
    "02834778": "bicycle",
    "03325088": "faucet",
    "04004475": "printer",
    "02954340": "cap"
}

reverse_mapping = {v: k for k, v in category_mapping.items()}


def get_args():
    return parser.parse_args()


def load_file(filelist):
    with open(filelist, 'r') as f:
        lines = f.read().split("\n")
    return lines


def get_json(filelist, top, categories, count):
    files = load_file(filelist)
    cats = {x.split("-")[0]: [] for x in files}
    for x in files:
        cats[x.split("-")[0]].append("-".join(x.split("-")[1:]))

    ckeys = sorted(cats, key=lambda k: len(cats[k]), reverse=True)
    if '' in ckeys:
        ckeys.remove('')
    jsondict = {}
    if categories[0] != 'A':
        ckeys = [x for x in ckeys if category_mapping[x] in categories]
        top = len(ckeys)
        print([category_mapping[x] for x in ckeys])
    for key in ckeys[:top]:
        jsondict[key] = cats[key][:count]

    return jsondict


def get_split(filelist, top, cats, total_count, train_ratio, test_ratio, val_ratio):
    jsondict = get_json(filelist, top, cats, total_count)

    final = {}

    train_ct = 0
    test_ct = 0
    val_ct = 0

    for key in jsondict:
        final[key] = {"train": [], "test": [], "val": []}
        final[key]["train"] = jsondict[key][:int(
            len(jsondict[key]) * train_ratio)]
        final[key]["test"] = jsondict[key][int(
            len(jsondict[key]) * train_ratio):int(len(jsondict[key]) * (train_ratio + test_ratio))]
        final[key]["val"] = jsondict[key][int(
            len(jsondict[key]) * (train_ratio + test_ratio)):]

        print(
            f"Key: {key}, Train: {len(final[key]['train'])}, Test: {len(final[key]['test'])}, Val: {len(final[key]['val'])}")

        train_ct += len(final[key]["train"])
        test_ct += len(final[key]["test"])
        val_ct += len(final[key]["val"])

    print(
        f"Top: {len(final.keys())}, Train: {train_ct}, Test: {test_ct}, Val: {val_ct}, Total: {train_ct + test_ct + val_ct}")

    return final


def save_jsons(final, dir, output):
    os.makedirs(dir, exist_ok=True)
    print(f"Saving to {dir}/{output}")
    with open(os.path.join(dir, output), 'w') as f:
        json.dump(final, f)

    # with open(os.path.join(dir, "train.json"), 'w') as f:
    #     json.dump(train, f)
    # with open(os.path.join(dir, "test.json"), 'w') as f:
    #     json.dump(test, f)
    # with open(os.path.join(dir, "val.json"), 'w') as f:
    #     json.dump(val, f)


if __name__ == "__main__":
    args = get_args()
    if args.train > 1 or args.test > 1 or args.val > 1 or args.train + args.test + args.val > 1 or args.train + args.test + args.val < 1:
        raise ValueError(
            "Train, test, and val ratios must be between 0 and 1 and must add up to 1")
    final = get_split(
        args.filelist, args.top, args.cats, args.count, args.train, args.test, args.val)
    save_jsons(final, args.dir, args.output)
