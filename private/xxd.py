# https://gist.github.com/rakete/c6f7795e30d5daf43bb600abc06a9ef4
import os.path
import string
import sys

def xxd(file_path, output_path):
    with open(file_path, 'r') as f:
        name = os.path.splitext(os.path.basename(file_path))
        array_name = name[0] + "_" + name[1][1:]
        output = "static uint8_t %s[] = {" % array_name
        length = 0
        while True:
            buf = f.read(12)

            if not buf:
                output = output[:-2]
                break
            else:
                output += "\n  "

            for i in buf:
                output += "0x%02x, " % ord(i)
                length += 1
        output += ", 0x00\n};\n"
        with open(output_path, "w") as output_file:
            output_file.write(output)


if __name__ == '__main__':
    if not os.path.exists(sys.argv[1]):
        print >> (sys.stderr, "The file doesn't exist.")
        sys.exit(1)
    xxd(sys.argv[1], sys.argv[2])
