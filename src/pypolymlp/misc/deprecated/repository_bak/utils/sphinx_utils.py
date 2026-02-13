#!/usr/bin/env python


def text_to_csv_table(
    data,
    title: str = None,
    header: list = None,
    widths: list = None,
    stub_columns=None,
    file=None,
):

    print(".. csv-table::", title, file=file)
    print("    :header:", header, file=file)
    print("    :widths:", end=" ", file=file)
    for w in widths[:-1]:
        print(w, end=",", file=file)
    print(widths[-1], file=file)

    if stub_columns is not None:
        print("    :stub-columns:", stub_columns, file=file)

    print("", file=file)
    for d in data:
        print("    ", end="", file=file)
        for d1 in d[:-1]:
            print(d1, end=",", file=file)
        print(d[-1], file=file)
    print("", file=file)
    print("", file=file)


def include_image(image, title=None, height=300, file=None):

    if title is not None:
        print("**" + title + "**", file=file)
        print("", file=file)

    print(".. image::", image, file=file)
    print("   :height: " + str(height) + "px", file=file)
    print("", file=file)
    print("", file=file)
