from PIL import Image
import json
import numpy as np
import cv2


#
# pil_img = Image.open('/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/original images/cologne_000148_000019_leftImg8bit.png')
# img = cv2.imread('/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/original images/cologne_000123_000019_leftImg8bit.png')
#
# print(pil_img.size)
#
# with open('/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/annotations/hamburg_000000_105123_gtBboxCityPersons.json') as jobject:
#     bbox = json.load(jobject)
#
# objects = bbox['objects']
# # print(objects)
# bboxes = [object_['bbox'] for object_ in objects if object_['label'] != 'ignore']
# print(bboxes)
#
#
# # print([obj['label'] for obj in objects])
#
#
# def grab_objects(bbox_):
#     _objects = bbox_['objects']
#     _bboxes = [object_['bboxVis'] for object_ in _objects if object_['label'] != 'ignore']
#     return np.array(bboxes)
#
#
# coordinates = grab_objects(bbox)
#
#
# def get_coordinates(coord):
#     coords_= coord
#     for coord in coords_:
#         x,y,w,h = coord
#         coord[2] = x + w
#         coord[3] = y + h
#     return coords_
#
#
# coords = get_coordinates(bboxes)
#
#
# def draw_bbox(_coordinates, img_, save=True):
#     for coord in _coordinates:
#         x0, y0, x1, y1 = coord
#         start_point = (int(x0), int(y0))
#         end_point = (int(x1), int(y1))
#         image = cv2.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=2)
#     if cv2.imwrite("example_with_bounding_boxes_.jpg", image):
#         print('Bounding boxes added successfully')
# #
# #
# draw_bbox(coords, img)

# ped1 = bboxes['objects'][12]
# bbox = ped1['bbox']

# print(len(bboxes['objects']))
#
def get_coordinates(coord):
    x, y, w, h = coord
    coord[2] = x + w
    coord[3] = y + h
    return coord


# #
#
# bbox = coordinates[1]
#
# box = get_coordinates(bbox)
#
# img2 = pil_img.crop(box)
# print(f'Image size: {img2.size}')
# print(box)
# print(coordinates)
#
# img2.show()

# def compute_area(image, ped_coords):
#     height, width = image.shape
#     area = height * width
#
#     area_taken = 0
#
#     for coord in ped_coords:
#         patch = image.crop(coord)
#         h, w = patch.size
#         a = h*w
#         area_taken += a
#     available_area = area - area_taken
#     print(f'The available space for cropping :{available_area}')
#     return available_area
#
#
# compute_area(pil_img, coords)
#
#
# def get_ped_count(annotation_path) -> '(np.array, max, min, ,mean)':
#     # this function return a np array with pedestrian count n for each image
#
#     count = []
#     annotations = os.listdir(annotation_path)
#
#     for annotation in annotations:
#         filepath = os.path.join(bbox_path, annotation)
#         with open(filepath) as jobject:
#             bounding_box = json.load(jobject)
#             objects = bounding_box['objects']
#             pedestrians = [obj['bboxVis'] for obj in objects if obj['label'] != 'ignore']
#             n = len(pedestrians)
#             count.append(n)
#
#     count = np.array(count)
#     max_count = count.max()
#     min_count = count.min()
#     avg_count = count.mean()
#     return count, max_count, min_count, avg_count


# ped_count = get_ped_count(bbox_path)
# print(f'print max number of pedestrians in an image : {ped_count.max()}')
# print(f'print lowest number of pedestrians in an image : {ped_count.min()}')
# print(f'print average number of pedestrians in an image : {ped_count.mean()}')

# print(ped_count)
#
# with open('bbox.json') as jobject:
#     bboxes = json.load(jobject)
# bboxes = bboxes['bbox']
#
#
# # c = [[6, 378, 25, 530], [-1808, 342, 1857, 462], [2034, 386, 2043, 438], [1967, 333, 20, 4],[1767, 358, 1818, 41]]
#
#
# def filter_bboxes(coordinates, threshold_area=300):
#     """This function is to discard any bounding boxes that have negative x,y coordinates and are below
#     a certain area i.e., h*w"""
#
#     x, y, w, h = coordinates
#     # if w * h >= threshold_area and x >= 0 and y >= 0 and x+w <= 2048 and y+h <= 1024:
#     if w * h >= threshold_area and x >= 0 and y >= 0 and x+w <= 2048 and y+h <= 1024:
#         return True
#     else:
#         return False
#
#
# for coord in bboxes:
#     if filter_bboxes(coord):
#         print(coord)


# img = [[189, 329, 28, 45], [1131, 343, 18, 45], [1455, 326, 23, 23], [1490, 332, 34, 34], [1508, 336, 11, 18],
#        [1519, 325, 20, 36], [1379, 306, 37, 88], [1438, 333, 21, 40], [1413, 331, 53, 81], [1452, 335, 46, 53],
#        [1527, 343, 33, 34], [1586, 332, 34, 25], [1503, 340, 60, 89], [1553, 340, 44, 87], [1693, 335, 27, 18],
#        [1683, 332, 11, 21], [1618, 339, 52, 89], [1657, 335, 48, 62], [1716, 347, 44, 23], [1717, 338, 39, 64],
#        [1802, 328, 31, 53], [1826, 329, 39, 52], [1832, 341, 32, 88], [1851, 339, 47, 48], [2008, 347, 26, 32],
#        [317, 368, 13, 20], [1276, 309, 56, 138], [-21, 326, 60, 146], [103, 343, 10, 23], [87, 343, 25, 53],
#        [61, 343, 43, 52], [155, 305, 32, 79], [272, 378, 59, 70], [151, 347, 20, 64], [246, 303, 41, 100],
#        [269, 335, 22, 20], [250, 336, 22, 32], [225, 338, 34, 46], [208, 349, 21, 19], [-9, 303, 85, 208],
#        [154, 352, 63, 110], [99, 302, 71, 174], [227, 371, 52, 80], [519, 331, 30, 73], [501, 331, 27, 75],
#        [558, 299, 46, 111], [350, 295, 69, 169], [283, 300, 67, 163], [465, 295, 58, 142], [457, 297, 60, 148],
#        [528, 313, 53, 130], [878, 336, 21, 50], [1071, 306, 59, 144], [949, 300, 64, 156], [1079, 359, 14, 33],
#        [325, 334, 54, 97], [355, 413, 5, 6], [393, 356, 19, 75], [419, 350, 12, 70], [428, 428, 4, 4],
#        [439, 389, 24, 42], [1586, 333, 12, 19], [1600, 331, 13, 16], [1761, 332, 45, 38]]
# img_coord = [get_coordinates(cd) for cd in img]
# r = [442, 211, 465, 269]
# r_coord = [442, 211, 907, 480]
#
#
# print(len(img))
#
# def is_intersect(random_bbox, image_bboxes):
#     rx, ry, rw, rh = random_bbox
#     i = 1
#
#     for bbox in image_bboxes:
#         x, y, w, h = bbox
#
#         rbbox_left_x = rx
#         rbbox_right_x = rx + rw
#         rbbox_top_y = ry
#         rbbox_bottom_y = ry + rh
#         # print(rbbox_left_x)
#         # print(rbbox_right_x)
#         # print(rbbox_top_y)
#         # print(rbbox_bottom_y)
#         # print()
#
#         ibbox_left_x = x
#         ibbox_right_x = x + w
#         ibbox_top_y = y
#         ibbox_bottom_y = y + h
#         # print(ibbox_left_x)
#         # print(ibbox_right_x)
#         # print(ibbox_top_y)
#         # print(ibbox_bottom_y)
#         # print()
#
#         if not (rbbox_right_x < ibbox_left_x or
#                 rbbox_left_x > ibbox_right_x or
#                 rbbox_top_y > ibbox_bottom_y or
#                 rbbox_bottom_y < ibbox_top_y):
#             return True
#         i += 1
#         print(i)
#
#     return False
#
#
# if is_intersect(r, img):
#     print('There is intersection')
# else:
#     print('There is no intersection')
#
#
#
    # def is_intersect(random_bbox, image_bboxes):
    #     rx, ry, rw, rh = random_bbox
    #     intersect = False
    #
    #     for bbox in image_bboxes:
    #         x, y, w, h = bbox
    #
    #         rbbox_left_x = rx
    #         rbbox_right_x = rx + rw
    #         rbbox_top_y = ry
    #         rbbox_bottom_y = ry + rh
    #
    #         ibbox_left_x = x
    #         ibbox_right_x = x + w
    #         ibbox_top_y = y
    #         ibbox_bottom_y = y + h
    #
    #         if not (rbbox_right_x < ibbox_left_x or
    #                 rbbox_left_x > ibbox_right_x or
    #                 rbbox_top_y > ibbox_bottom_y or
    #                 rbbox_bottom_y < ibbox_top_y):
    #             intersect = True
    #
    #     return intersect


'cologne_000018_000019'