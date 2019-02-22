from __future__ import division

from PIL import Image
import os,sys
import numpy as np


def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


def check_win(win):
    '''
    Input:
    	win: window for current node
    Output:
    	val: value of current window
    '''
    if win.min() == win.max():
        return win.min()
    else:
 	      return float(-1)


def node_val(raw_map, given_level, total_levels, key):
    '''
    Input:
    	raw_map: original segmentation map
    	level: depth level of segmentation map
    	key: hash-table key (x,y)
    Output:
    	val: value of current node
    '''
    x = key[0]
    y = key[1]
    im_x = raw_map.shape[0]
    im_y = raw_map.shape[1]
    win_size = np.power(2,(total_levels - given_level))
    win_x_min = int(max(x*win_size, 0))
    win_y_min = int(max(y*win_size, 0))
    win_x_max = int(min((x+1)*win_size, im_x))
    win_y_max = int(min((y+1)*win_size, im_y))
    win = raw_map[win_x_min:win_x_max,win_y_min:win_y_max]
    val = check_win(win)
    return val


def cur_key_to_last_key(key):
    '''
    Input:
    	key: (x,y) cordinates of current pixel
    Output:
    	last_key: (x,y) cordinates of super-pixel for current pixel
    '''
    last_key = (int(key[0]/2), int(key[1]/2))
    return last_key


def node_val_inter(raw_map, given_level, total_levels, key, last_map):
    '''
    Input:
            raw_map: original segmentation map
            level: depth level of segmentation map
            key: hash-table key (x,y)
    Output:
            val: value of current node
    '''
    last_key = cur_key_to_last_key(key)
    last_x = last_key[0]
    last_y = last_key[1]
    last_val = last_map[last_x,last_y]
    if last_val >= 0:
        return last_val
    x = key[0]
    y = key[1]
    im_x = raw_map.shape[0]
    im_y = raw_map.shape[1]
    win_size = np.power(2,(total_levels - given_level))
    win_x_min = int(max(x*win_size, 0))
    win_y_min = int(max(y*win_size, 0))
    win_x_max = int(min((x+1)*win_size, im_x))
    win_y_max = int(min((y+1)*win_size, im_y))
    win = raw_map[win_x_min:win_x_max,win_y_min:win_y_max]
    val = check_win(win)
    return val


def depth_sub(previous_map, current_map):
  	'''
  	Input:
  		previous_map: seg map of last layer
  		current_map: seg map of current layer
  	Output:
  		current_map: current_map - scaled(previous_map)
  	'''
  	current_map_x = current_map.shape[0]
  	current_map_y = current_map.shape[1]
  	previous_map = Image.fromarray(np.array(previous_map, dtype=np.uint8)).resize(size=(current_map_x,current_map_y), resample=Image.NEAREST)
  	previous_map = np.array(previous_map)
  	return current_map - previous_map


def dense2quad(raw_map, num_levels=6):
    '''
    raw_map: input is raw segmentation map
    out_map: output is quadtree output representation
    '''
    size_x = raw_map.shape[0]
    size_y = raw_map.shape[1]
    
    init_res_x = int(size_x/np.power(2,num_levels))
    init_res_y = int(size_y/np.power(2,num_levels))
    
    out_map = {}
    for given_level in range(1,num_levels+1):
        level_res_x = init_res_x*np.power(2,given_level)
        level_res_y = init_res_y*np.power(2,given_level)
        level_map = np.zeros((level_res_x,level_res_y), dtype=np.float32)
        for x in range(0,level_res_x):
            for y in range(0,level_res_y):
        		    if given_level == 1: level_map[x,y] = node_val(raw_map, given_level, num_levels, (x,y))
        		    else: level_map[x,y] = node_val_inter(raw_map, given_level, num_levels, (x,y), out_map[given_level-1])
        out_map[given_level] = level_map
    return out_map


if __name__ == '__main__':

    data_root = '../data/SUNRGBD/label37/'
    data_type = 'train'
    
    data_dir = data_root + data_type + '/'
    list_files = [data_dir + f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    
    # data_root = '../../Cityscapes/gtFine_trainvaltest/gtFine/'
    # data_type = 'val'
    #
    # data_dir = data_root + data_type + '/'
    # list_files = []
    # for city in os.listdir(data_dir):
    # 	data_dir_new = data_dir + city + '/'
    # 	list_files += [data_dir_new + f for f in os.listdir(data_dir_new) if (os.path.isfile(os.path.join(data_dir_new, f)) and f.endswith('labelIds.png'))]
    
    num_levels = 6
    valid_pixels_at_level = np.zeros(num_levels)
    total_pixels_at_level = np.zeros(num_levels)
    for i, filename in enumerate(list_files):
      	segm = np.array(Image.open(filename))
      	print(i, segm.shape)
      	segm_rounded_height = round2nearest_multiple(segm.shape[0], np.power(2,num_levels-1))
      	segm_rounded_width = round2nearest_multiple(segm.shape[1], np.power(2,num_levels-1))
      	segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
      	segm_rounded[:segm.shape[0], :segm.shape[1]] = segm
      	out_map = dense2quad(segm_rounded, num_levels)
      	for given_level in range(1,num_levels+1):
            this_level_count = np.sum(out_map[given_level]!=-1)
            this_level_total = (out_map[given_level].shape[0]*out_map[given_level].shape[1])
            valid_pixels_at_level[given_level-1] += this_level_count
            total_pixels_at_level[given_level-1] += this_level_total
    
    ratios = valid_pixels_at_level/total_pixels_at_level
    print(ratios[0])
    for i in range(1,num_levels):
   	    print(ratios[i]-ratios[i-1])
