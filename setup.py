from setuptools import setup, find_packages

setup(name="aicity_2024_driving_action", version="1.0", author="TRAN DAI CHI", author_email="ctran743@gmail.com", description="README.md", url="",
      py_modules=["UniformerV2_1_train", "UniformerV2_2_train", "VideoMAE_train", "X3D_train", "demo", "eval", "exp", "infer"],
      license="LICENSE", python_requires=">=3.8", include_package_data=True, install_requires="requirements.txt")