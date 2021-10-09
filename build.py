import os 
import mesonbuild

def build(setup_kwargs):
	os.system("c++ --version")
	os.system("python -m mesonbuild.mesonmain build")
	os.system("sudo meson compile -vC build")
	os.system("sudo meson install -C build")
	print("\n==== Finished meson build ====\n")