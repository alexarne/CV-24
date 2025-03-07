import cv2
import numpy as np
from utils import generate_2d_points, extract_and_match_SIFT
from plots import draw_homography,draw_matches

def find_homography(pts1:np.ndarray, pts2:np.ndarray) -> np.ndarray:
    '''Find the homography matrix from matching points in two images.

        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :returns np.ndarray H: a 3x3 array representing the homography matrix H.

    '''

    # #TODO: ADD YOUR CODE HERE
    # raise NotImplementedError("Homography estimation yet to be done. Complete the function find_homography and remove this line.")

    y1, x1 = pts1.shape
    y2, x2 = pts2.shape
    if x1 != x2:
        raise AssertionError("Homography: Input vectors not equal size.")
    if x1 < 4:
        raise AssertionError("Homography: Input vectors size less than 4.")
    
    A = []
    for (x_a, y_a), (x_b, y_b) in zip(np.transpose(pts1), np.transpose(pts2)):
        A.append([x_a, y_a, 1, 0, 0, 0, -x_a*x_b, -y_a*x_b, -x_b])
        A.append([0, 0, 0, x_a, y_a, 1, -x_a*y_b, -y_a*y_b, -y_b])
    A = np.array(A)

    U, S, Vh = np.linalg.svd(A)
    h = Vh[8, :]
    H = h.reshape(3, 3)
    H = H / H[-1, -1]
    return H


def homography_error(H1:np.ndarray, H2:np.ndarray, focal:float = 1000) -> float:
    '''Computes the error between two homographies, wrt a known focal.
        :param np.ndarray H1: a 3x3 matrix representing one of the homographies.
        :param np.ndarray H2: a 3x3 matrix representing the second homography.
        :param float focal: the known focal length.
        :returns float: the error between the homographies.
    '''
    H_diff = H1/H1[2,2] - H2/H2[2,2]
    return np.linalg.norm(np.diag((1/focal,1/focal,1)) @ H_diff @ np.diag((focal,focal,1)))


def count_homography_inliers(H:np.ndarray, pts1:np.ndarray, pts2:np.ndarray, thresh:float = 1.0) -> tuple[int,np.ndarray]:
    '''Given the homography H, projects pts1 on the second image, counting the number of actual points in pts2 for which the projection error is smaller than the given threshold.

        :param np.ndarray H: a 3x3 matrix containing the homography matrix.
        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :param float thresh: the threshold to consider points as inliers.
        :returns int ninliers, np.ndarray errors:
            ninliers: the number of inliers.
            errors: a N_points array containing the errors; they are indexed as pts1 and pts2.
    
    '''
    Hp1 = H @ np.vstack((pts1, np.ones((1, np.size(pts1, axis=1)))))
    errors = np.sqrt(np.sum((Hp1[0:2,:]/Hp1[2,:] - pts2)**2, axis=0))
    ninliers = np.sum(np.where(errors<thresh**2, 1, 0))
    return ninliers, errors


def find_homography_RANSAC(pts1:np.ndarray, pts2:np.ndarray, niter:int = 100, thresh:float = 1.0) ->tuple[np.ndarray,int,np.ndarray]:
    '''Computes the best homography for matching points pts1 and pts2, adopting RANSAC.

        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :param int niter: the number of RANSAC iteration to run.
        :param float thresh: the maximum error to consider a point as an inlier while evaluating a RANSAC iteration.
        :returns np.ndarray Hbest, int ninliers, np.ndarray errors:
            Hbest: a 3x3 matrix representing the best homography found.
            ninliers: the number of inliers for the best homography found.
            errors: a N_points array containing the errors for the best homography found; they are indexed as pts1 and pts2.
    
    '''

    # #TODO: ADD YOUR CODE HERE
    # raise NotImplementedError("RANSAC Based homography yet to be done. Complete the function find_homography_RANSAC and remove this line.")

    y1, x1 = pts1.shape
    y2, x2 = pts2.shape
    if x1 != x2:
        raise AssertionError("Homography: Input vectors not equal size.")
    if x1 < 4:
        raise AssertionError("Homography: Input vectors size less than 4.")
    N = x1

    ninliers = 0
    for i in range(niter):
        indices = np.random.choice(N, 4, replace=False)
        set1 = pts1[:, indices]
        set2 = pts2[:, indices]
        H = find_homography(set1, set2)

        local_ninliers, local_errors = count_homography_inliers(H, pts1, pts2, thresh)
        if local_ninliers > ninliers:
            Hbest = H
            ninliers = local_ninliers
            errors = local_errors
            print(f"Inliers {ninliers} at iteration {i}")

    return Hbest, ninliers, errors


def synthetic_example(RANSAC = False):
    focal = 1000
    pts1, pts2, H = generate_2d_points(num = 100, noutliers = 50, noise=0.1, focal = focal)
    draw_matches(pts1, pts2)
    print('True H =\n', H)
    if RANSAC:
        H1,ninliers,errors = find_homography_RANSAC(pts1, pts2, niter=10000)
        H2 = find_homography(pts1[:,errors<1], pts2[:,errors<1])
        print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
        print('RANSAC H =\n', H1)
        print('Final estimated H =\n', H2)
    else:
        H2 = find_homography(pts1, pts2)
        print('Estimated H =\n', H2)
    print('Error =', homography_error(H, H2, focal))


def real_example():
    # img1 = cv2.imread('images/img1.jpg', 0)
    # img2 = cv2.imread('images/img2.jpg', 0)
    img1 = cv2.imread('images/books1.jpg', 0)
    img2 = cv2.imread('images/books2.jpg', 0)
    pts1, pts2 = extract_and_match_SIFT(img1, img2, num = 1000)
    H1,ninliers,errors = find_homography_RANSAC(pts1, pts2, niter=10000)
    H2 = find_homography(pts1[:,errors<1], pts2[:,errors<1])
    print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
    print('RANSAC H =\n', H1)
    print('Final estimated H =\n', H2)
    draw_homography(img1, img2, H2)


if __name__=="__main__":
    np.set_printoptions(precision = 3)

    #TODO: You can use this function to perform your tests or try our examples, uncommenting them

    ## Task 1 example
    # synthetic_example(RANSAC = False)

    ## Task 2 example (from synthetic data)
    synthetic_example(RANSAC = True)

    ## Task 2 example (from real images)
    # real_example()