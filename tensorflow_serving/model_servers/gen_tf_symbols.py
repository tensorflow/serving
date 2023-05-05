import os
import sys


def parse_args(argv):
  result = {}
  for arg in argv:
    k, v = arg.split("=")
    result[k] = v

  return result


def generate_version(header_in, header_out):
  header_out = os.path.expanduser(header_out)
  header_out_dir = os.path.dirname(header_out)

  if not os.path.exists(header_out_dir):
    os.makedirs(header_out_dir, exist_ok=True)

  with open(header_out, "w") as outf:
    outf.write("{\n")
    with open(os.path.expanduser(header_in)) as inf:
      content = inf.readlines()

      for line in content:
        line = line.strip().strip(";")

        # Excluding below symbols to avoid link warining.
        if line.find("lite") != -1 or line.find("tsl") != -1:
          continue
        outf.writelines("    " + line + ";\n")

      outf.write("};")


def main():
  args = parse_args(sys.argv[1:])
  generate_version(args["--in"], args["--out"])


if __name__ == "__main__":
  main()
