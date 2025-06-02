import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
# import cv2  # 添加 OpenCV 库
import numpy as np

def read_aes_key(file_path: str) -> bytes:
    """从文件中读取AES密钥，假设文件内容为16或32字节的二进制数据"""
    with open(file_path, 'rb') as f:  # 以二进制方式打开密钥文件
        key = f.read()  # 读取文件内容到key变量
    if len(key) not in (16, 32):  # 检查密钥长度是否为16或32字节
        raise ValueError("AES key must be 16 or 32 bytes long")  # 如果不是，抛出异常
    return key  # 返回读取到的密钥

# 预共享密钥（16字节=128位，或32字节=256位）
try:
    key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "aes.key")
    # 读取AES密钥，支持多操作系统的路径拼接
    AES_KEY = read_aes_key(str(key_path))
except FileNotFoundError:
    raise FileNotFoundError(f"AES key file not found: {key_path}")
except ValueError as ve:
    raise ValueError(f"AES key read error: {ve}")
except Exception as e:
    raise RuntimeError(f"Unknown error occurred while reading AES key: {e}")

def generate_iv():
    """生成16字节的随机IV"""
    return os.urandom(16)

def aes_encrypt_cbc(plaintext: bytes, key: bytes) -> bytes:
    """使用AES-CBC模式加密数据，返回IV+密文"""
    iv = generate_iv()  # 生成16字节的随机IV
    padder = padding.PKCS7(128).padder()  # 创建PKCS7填充器，块大小为128位
    padded_data = padder.update(plaintext) + padder.finalize()  # 对明文进行填充
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())  # 创建AES-CBC加密器
    encryptor = cipher.encryptor()  # 获取加密器对象
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()  # 加密填充后的数据
    return iv + ciphertext  # 返回IV和密文拼接后的结果

def aes_decrypt_cbc(iv_ciphertext: bytes, key: bytes) -> bytes:
    """使用AES-CBC模式解密数据，输入为IV+密文，返回明文"""
    iv = iv_ciphertext[:16]  # 提取前16字节作为IV
    ciphertext = iv_ciphertext[16:]  # 剩余部分为密文
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())  # 创建AES-CBC解密器
    decryptor = cipher.decryptor()  # 获取解密器对象
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()  # 解密密文得到填充后的明文
    unpadder = padding.PKCS7(128).unpadder()  # 创建PKCS7去填充器，块大小为128位
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()  # 去除填充，得到原始明文
    return plaintext  # 返回明文

def calculate_psnr(original_frame, decoded_frame):
    """
    计算两帧之间的 PSNR 值。
    Args:
        original_frame (numpy.ndarray): 原始帧。
        decoded_frame (numpy.ndarray): 解码后的帧。
    Returns:
        float: PSNR 值（以分贝为单位）。
    """
    mse = np.mean((original_frame - decoded_frame) ** 2)
    if mse == 0:
        return float('inf')  # 如果 MSE 为 0，PSNR 为无穷大
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# 示例用法
# if __name__ == "__main__":
#     # 明文数据
#     segment_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "video_segments", "bbb_sunflower", "480p-1500k", "bbb_sunflower-480p-1500k-000.ts")
#     with open(segment_path, "rb") as f:
#         data = f.read()
#     # 加密
#     encrypted = aes_encrypt_cbc(data, AES_KEY)
#     print(f"Encrypted (IV+data): {encrypted.hex()}")
#     # 解密
#     decrypted = aes_decrypt_cbc(encrypted, AES_KEY)
#     # 将解密后的视频分片以.ts文件格式写入当前文件夹
#     with open("decrypted_segment.ts", "wb") as f:
#         f.write(decrypted)
#     print(f"Decrypted: {decrypted}")

#     # 解密后计算 PSNR
#     original_video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "original_video.mp4")
#     cap_original = cv2.VideoCapture(original_video_path)
#     cap_decrypted = cv2.VideoCapture("decrypted_segment.ts")

#     while cap_original.isOpened() and cap_decrypted.isOpened():
#         ret_orig, frame_orig = cap_original.read()
#         ret_dec, frame_dec = cap_decrypted.read()
#         if not ret_orig or not ret_dec:
#             break
#         psnr_value = calculate_psnr(frame_orig, frame_dec)
#         print(f"PSNR: {psnr_value:.2f} dB")

#     cap_original.release()
#     cap_decrypted.release()
