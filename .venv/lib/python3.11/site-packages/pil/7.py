import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
from PIL import Image
import io
import os

img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

def huffman_encoding(data):
    freq = defaultdict(int)
    for value in data:
        freq[value] += 1
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    huff_dict = dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))
    return huff_dict

flat = img.flatten()
huff_dict = huffman_encoding(flat)
original_bits = len(flat) * 8
encoded_bits = sum([len(huff_dict[val]) for val in flat])
compression_ratio = original_bits / encoded_bits
print(f"Huffman Compression Ratio: {compression_ratio:.2f}")

def run_length_encode(data):
    encoding = []
    prev_pixel = data[0]
    count = 1
    for pixel in data[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            encoding.append((prev_pixel, count))
            prev_pixel = pixel
            count = 1
    encoding.append((prev_pixel, count))
    return encoding

def run_length_decode(encoding):
    decoded = []
    for val, count in encoding:
        decoded.extend([val] * count)
    return np.array(decoded)

rle_encoded = run_length_encode(flat)
rle_decoded = run_length_decode(rle_encoded).reshape(img.shape)

plt.subplot(1, 3, 2)
plt.imshow(rle_decoded, cmap='gray')
plt.title("RLE Decompressed Image")
plt.axis('off')

print(f"Original Size: {len(flat)} pixels")
print(f"RLE Encoded Size: {len(rle_encoded)} runs")

cv2.imwrite("compressed.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
jpeg_img = cv2.imread("compressed.jpg", cv2.IMREAD_GRAYSCALE)

plt.subplot(1, 3, 3)
plt.imshow(jpeg_img, cmap='gray')
plt.title("JPEG Compressed (Lossy)")
plt.axis('off')

plt.tight_layout()
plt.show()

orig_size = os.path.getsize("sample.jpg")
jpeg_size = os.path.getsize("compressed.jpg")
print(f"Original File Size: {orig_size/1024:.2f} KB")
print(f"JPEG Compressed Size: {jpeg_size/1024:.2f} KB")
print(f"JPEG Compression Ratio: {orig_size/jpeg_size:.2f}")