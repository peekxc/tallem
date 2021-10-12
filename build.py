import os 
import sys 
from pathlib import Path
import mesonbuild

# From: https://stackoverflow.com/questions/51108256/how-to-take-a-pathname-string-with-wildcards-and-resolve-the-glob-with-pathlib
def expandpath(path_pattern):
	p = Path(path_pattern).expanduser()
	parts = p.parts[p.is_absolute():]
	return Path(p.root).glob(str(Path(*parts)))

def build(setup_kwargs):
	suffix = os.popen('python3-config --extension-suffix').read().rstrip()
	print(f"Building extensions with suffix: {suffix}")
	existing_modules = list(expandpath(f"~/tallem/src/tallem/pbm/*{suffix}"))
	if len(existing_modules) > 0:
		print("Removing existing modules for a clean build")
	#ext_modules = [p.name for p in ]
	
	## Remove existing extension modules
	for m in existing_modules:
		os.remove(m)

	## Recompile
	print("\n==== Printing compiler version ====\n")
	os.system("c++ --version")
	print("\n==== Starting meson build ====\n")
	os.system("meson setup build")
	os.system("meson compile -vC build")
	target_path = next(expandpath("~/tallem/src/tallem/pbm/")).resolve()
	print(f"Install path: {target_path}")
	os.system(f"sudo cp build/*{suffix} {target_path}")
	# os.system("meson install -C build")
	print("\n==== Finished meson build ====\n")
	
	## Check if they now exist
	num_so = len([p.name for p in expandpath(f"~/tallem/src/tallem/pbm/*{suffix}")])
	if num_so > 0:
		return(0)
	else: 
		print("Did not detect native python extensions; Exiting build")
		sys.exit(-1)
