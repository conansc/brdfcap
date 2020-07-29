import matplotlib.pyplot as plt
import numpy as np
import os


def draw_highlighted_points(img, points, color=[0, 0, 255], circ=0, inv=False):
    """
    Highlights points in an image
    :param img: Input image to highlight in
    :param points: Coordinates of points to highlight
    :param color: Color for highlighting
    :param circ: Size of area to highlight
    :param inv: Swap x and y coordinates
    :return: Image with highlighted points
    """
    for point in points:
        img = draw_highlighted_point(img, point, color, circ, inv)
    return img


def draw_highlighted_point(img, point, color, circ, inv):
    """
    Highlights a point in an image
    :param img: Input image to highlight in
    :param point: Coordinates of point to highlight
    :param color: Color for highlighting
    :param circ: Size of area to highlight
    :param inv: Swap x and y coordinates
    :return: Image with highlighted point
    """
    height, width, _ = img.shape
    color = np.array(color)
    point = np.squeeze(point)

    if circ > 0:
        ran = range(-circ, circ)
    else:
        ran = [0]

    if inv:
        x = int(point[1])
        y = int(point[0])
    else:
        x = int(point[0])
        y = int(point[1])

    for i in ran:
        for j in ran:
            curr_y = y + j
            curr_x = x + i
            if height <= curr_y < 0 or width <= curr_x < 0:
                continue
            img[curr_y, curr_x, :] = color

    return img


def plot_brdf(path, name, ills, theta_ins, theta_outs, phi_diffs, cfa, axis_names, c_names, only_perfect=False):
    """
    Plots BRDF values
    :param path: Path to save image in
    :param name: Name of file to save
    :param ills: Illumination values
    :param theta_ins: Theta in values
    :param theta_outs: Theta out values
    :param phi_diffs: Phi diff values
    :param cfa: CFA values
    :param axis_names: Name of axes in plot
    :param c_names: Name of colors per CFA channel
    :param only_perfect: Plot only perfect reflection (theta_in == theta_out)
    :return: None
    """
    xs = theta_ins - theta_outs
    xs /= np.pi
    xs *= 180
    ys = ills

    phi_diffs /= np.pi
    phi_diffs *= 180
    phi_diffs -= 180

    ys = np.array([y for _, y in sorted(zip(xs, ys))])
    phi_diffs = np.array([pd for _, pd in sorted(zip(xs, phi_diffs))])
    xs = np.array(sorted(xs))

    if only_perfect:
        ids1 = np.where(phi_diffs > -2)
        ids2 = np.where(phi_diffs < 2)
        ids = np.intersect1d(ids1, ids2)
        xs = xs[ids]
        ys = ys[ids]

    fig = plt.figure()

    c_ids = np.unique(cfa)
    for i in range(len(c_ids)):
        c_idx = c_ids[i]
        c_name = c_names[i]
        ax = fig.add_subplot(221 + c_idx)
        ax.scatter(xs[cfa == c_idx], ys[cfa == c_idx], color=c_name, s=0.1, alpha=0.7)
        ax.set_xlim(-25, 25)

    fig.suptitle(axis_names[2])
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    plt.savefig(os.path.join(path, name + ".png"), dpi=400)
    plt.close()
