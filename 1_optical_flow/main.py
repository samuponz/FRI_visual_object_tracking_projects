# --------------------------------------------------------------------------------------------
from ex1_utils import read_images, synthetic_images_generation
from of_methods import lucaskanade, hornschunck, show_lk_results, show_hs_results, show_both_results, show_flow
from matplotlib import pyplot as plt


if __name__ == '__main__':

    # Synthetic image test
    kernel_size = 3; cond_th = 1e2 ; low_th = 1e-3  # LK params
    n_iter = 1000; lmbd = 0.5  # HS params
    im1, im2 = synthetic_images_generation()
    u_lk1, v_lk1, mask_lk = lucaskanade(im1, im2, 3, cond_th, low_th)
    u_hs, v_hs = hornschunck(im1, im2, n_iter, lmbd)

    # Real image 1 test 1
    # kernel_size = 5; cond_th = 1e2; low_th = 1e-1  # LK params
    # n_iter = 1000; lmbd = 0.5  # HS params
    # prev = 'disparity/cporta_left.png'; succ = 'disparity/cporta_right.png'
    # im1, im2 = read_images(prev,succ)
    # u_lk, v_lk, mask = lucaskanade(im1, im2, kernel_size, cond_th, low_th)
    # u_hs, v_hs = hornschunck(im1, im2, n_iter, lmbd)
    # show_both_results(im1, im2, u_lk, v_lk, u_hs, v_hs)

    # Real image 2 test 1
    # kernel_size = 3; cond_th = 1e2; low_th = 1e-3  # LK params
    # n_iter = 1000; lmbd = 0.5  # HS params
    # prev = 'disparity/office2_left.png'; succ = 'disparity/office2_right.png'
    # im1, im2 = read_images(prev, succ)
    # u_lk, v_lk, mask = lucaskanade(im1, im2, kernel_size, cond_th, low_th)
    # u_hs, v_hs = hornschunck(im1, im2, n_iter, lmbd)
    # show_both_results(im1, im2, u_lk, v_lk, u_hs, v_hs)

    # Real image 3 test 1
    # kernel_size = 3; cond_th = 1e2; low_th = 1e-3  # LK params
    # n_iter = 1000; lmbd = 0.5  # HS params
    # prev = 'disparity/office_left.png'; succ = 'disparity/office_right.png'
    # im1, im2 = read_images(prev, succ)
    # u_lk, v_lk, mask = lucaskanade(im1, im2, kernel_size, cond_th, low_th)
    # u_hs, v_hs = hornschunck(im1, im2, n_iter, lmbd)
    # show_both_results(im1, im2, u_lk, v_lk, u_hs, v_hs)

    plt.show()
