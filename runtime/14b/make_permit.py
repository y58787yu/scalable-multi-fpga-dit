"""Read role receipts as a JSON list on stdin and write a validated cluster permit."""
import argparse
import json
import sys
from run import configure_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", required=True, type=int)
    parser.add_argument("--phase", required=True, choices=("INITIALIZE", "RELEASE", "RESUME"))
    args = parser.parse_args()
    configure_groups(args.groups)
    from sp3_control import permit
    json.dump(permit(json.load(sys.stdin), args.phase), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
