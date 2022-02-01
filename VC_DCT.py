import numpy as np
import cv2 
import math
import itertools

jpeg_quantiz_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])

def pieces(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def zig_zag(array, n=None):

    shape = np.array(array).shape
    res = np.zeros_like(array)

    j = 0
    i = 0
    direction = 'r'  # {'r': right, 'd': down, 'ur': up-right, 'dl': down-left}
    
    for count in range(1, n + 1):
        
        res[j][i] = array[j][i]
        
        if direction == 'r':
            i += 1
            if j == shape[0] - 1:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'dl':
            i -= 1
            j += 1
            if j == shape[0] - 1:
                direction = 'r'
            elif i == 0:
                direction = 'd'
        elif direction == 'd':
            j += 1
            if i == 0:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'ur':
            i += 1
            j -= 1
            if i == shape[1] - 1:
                direction = 'd'
            elif j == 0:
                direction = 'r'
    
    #print('done2')
    return res

        
def compress(image,num_coeffs=None,scale_factor=1):

    image = np.float32(image)

    h, w = image.shape

    n_height = np.int32(math.ceil(h / 8)) * 8
    n_width = np.int32(math.ceil(w / 8)) * 8

    new_canvas = np.zeros((n_height, n_width))
    new_canvas[0:h, 0:w] = image

    image = np.float32(new_canvas)
    height, width = image.shape

    image_blocks = [image[j:j + 8, i:i + 8] for (j, i) in itertools.product(range(0, height, 8),range(0, width, 8))]

    # Applying DCT for every block
    dct_blocks = [cv2.dct(image_block) for image_block in image_blocks]
    
    if num_coeffs is not None:
        # Keep only the first K DCT coefficients of every block
        reduced_dct_coeffs = [zig_zag(dct_block, num_coeffs) for dct_block in dct_blocks]
        
    else:
        # Quantize all the DCT coefficients using the quantization matrix and the scaling factor
        reduced_dct_coeffs = [np.round(dct_block / (jpeg_quantiz_matrix * scale_factor)) for dct_block in dct_blocks]
        reduced_dct_coeffs = [reduced_dct_coeff * (jpeg_quantiz_matrix * scale_factor) for reduced_dct_coeff in reduced_dct_coeffs]

    comp_image_blocks = [cv2.idct(coeff_block) for coeff_block in reduced_dct_coeffs]

    comp_image = []
    
    for chunk_row_blocks in pieces(comp_image_blocks, width//8):
        for row_block_num in range(8):
            for block in chunk_row_blocks:
                comp_image.extend(block[row_block_num])
                
    comp_image = np.array(comp_image).reshape(height, width)

    # round to the nearest integer [0,255] value
    comp_image[comp_image < 0] = 0
    comp_image[comp_image > 255] = 255
    comp_image = np.uint8(comp_image)

    return comp_image[0:h, 0:w]


def main():
    input_video = cv2.VideoCapture('cv.mp4')
    if input_video.isOpened()==False:
        print("Error in opening video")
    count = 0
    compressed_frames = []
    while input_video.isOpened():       
        ret, frame = input_video.read()
        if ret==True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            compressed_frames.append(compress(frame,4))
            
            if count==0:
                cv2.imwrite('firstframe_in.jpg',frame)
                cv2.imwrite('firstframe_out.jpg',compress(frame,4))
            
            print('Compressed Frame: ',count)
            #break
            count+=1 
        else:
            break
        
    #print(count) #349,593   
    h,w = compressed_frames[0].shape    
    fourcc= cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter('output.mp4',fourcc,15,(h,w))
    
    for frame in compressed_frames:
        output_video.write(frame)
    
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()
            
main()