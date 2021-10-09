import os 
import mesonbuild

def build(setup_kwargs):
	print("\n==== Printing compiler version ====\n")
	os.system("c++ --version")
	print("\n==== Starting meson build ====\n")
	os.system("meson build")
	os.system("meson compile -vC build")
	os.system("meson install -C build")
	print("\n==== Finished meson build ====\n")