import os
from tabnanny import verbose
import numpy as np
import cvxpy as cp
import cv2 as cv
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


def random_face(feat_mat: np.ndarray, dim: int) -> np.ndarray:
    kernel = np.random.normal(size=[feat_mat.shape[1], dim])
    kernel = normalize(kernel)
    return feat_mat @ kernel


def cone_programming(feat_mat: np.ndarray, input: np.ndarray, epsilon: int) -> np.ndarray:
    lin_para = cp.Variable(feat_mat.shape[0])
    cons = cp.SOC(epsilon, lin_para @ feat_mat - input)
    obj_func = cp.Minimize(cp.norm(lin_para, p=1))
    prob = cp.Problem(obj_func, [cons])
    prob.solve()
    return lin_para.value


def sc_index(k: int, num_per_id: int, x: np.ndarray):
    l1_norm = np.sum(np.abs(x))
    ratio = 0
    for i in range(k):
        this_ratio = np.sum(np.abs(x[i*num_per_id:(i+1)*num_per_id]))
        this_ratio /= l1_norm
        ratio = np.max([ratio, this_ratio])
    sc_idx = (k*ratio - 1) / (k-1)
    return sc_idx


def id_reco(k: int, id_range, x: np.ndarray):
    pass


def main():
    threshold = 0.6


if __name__ == "__main__":
    nomask_path = "D:\\Code_uc\\face_reco\\output\\dataset_nomask"
    mask_path = "D:\\Code_uc\\face_reco\\output\\dataset_mask"
    X_nomask = []
    for i, subject_name in enumerate(os.listdir(mask_path)):
        subject_images_dir = os.path.join(mask_path, subject_name)
        for img_name in os.listdir(subject_images_dir):
            img_path = os.path.join(subject_images_dir, img_name)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            img = np.reshape(img, [-1])
            X_nomask.append(img)

    feat_size = 40
    X_nomask = np.asarray(X_nomask)
    pca = PCA(n_components=feat_size)
    pca.fit(X_nomask.T)
    feat_mat = pca.components_.T

    X_train = np.ndarray(shape=[0, feat_size])
    X_test_ac = np.ndarray(shape=[0, feat_size])
    y_test_id = []
    num_per_id = 11
    num_training_cls = 40
    for i in range(num_training_cls):
        X_train = np.vstack([X_train, feat_mat[15*i:15*i+num_per_id]])
        X_test_ac = np.vstack([X_test_ac, feat_mat[15*i+num_per_id:15*i+15]])
        y_test_id.append([i] * (15-num_per_id))
    X_test_rj = feat_mat[15*num_training_cls:]

    sc_idx_ac = []
    sc_idx_rj = []
    epsilon = 1e-4
    tao = 0.15
    for rj_input in X_test_ac:
        para = cone_programming(X_train, rj_input, epsilon)
        sc_idx_ac.append(sc_index(num_training_cls, num_per_id, para))
    for rj_input in X_test_rj:
        para = cone_programming(X_train, rj_input, epsilon)
        sc_idx_rj.append(sc_index(num_training_cls, num_per_id, para))
    sc_idx_ac, sc_idx_rj = np.asarray(sc_idx_ac), np.asarray(sc_idx_rj)
    ac_num = np.count_nonzero(sc_idx_ac > tao)
    rj_num = np.count_nonzero(sc_idx_rj < tao)
    print(ac_num, rj_num)
