import MNIST_Loader as load
import Network as Net
import sys
import getopt

def help():
    print "\nThis script is used to test and train the Network class with the MNIST dataset.\n" \
          "It can also be used to classify provided handwritten numbers using its training from the MNIST set.\n" \
          "\n The following options are available: \n" \
          "     -h (help)     : Will display this output \n" \
          "     -t (train)    : Set 1 or 0 to indicate whether or not you want to train it again. \n" \
          "     -v (visual)   : Set 1 or 0 to indicate whether or not you want a visual test. \n" \
          "     -f (filename) : Specify a file patch after this of a file to test (must train first, set the name of " \
          "the file to be the correct number).\n\n"

def main(argv):
    # Get options specified
    try:
        options, args = getopt.getopt(argv, "ht:v:f:", ["help", "train", "visual", "filename"])
    except getopt.GetoptError:
        print "\n Invalid argument(s)."
        help()
        sys.exit(2)

    train    = 0
    visual   = 0
    filename = None

    for o, a in options:
        if o in ("-h", "--help"):
            help()
            sys.exit()

        elif o in ("-t", "--train"):
            train = a

        elif o in ("-v", "--visual"):
            visual = a

        elif o in ("-f", "--filename"):
            filename = a

    training_data, val_data, test_data = load.load_data_wrapper()
    net = Net.Network([794, 30, 10])

    if filename:
        print filename
        try:
            image = load.load_custom_image(filename, filename[0])
        except:
            print "\n Error loading image, check path again, ensure no quotes used.\n"
            sys.exit(2)
    else:
        image = test_data

    if train:
        net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    if visual:
        net.visual_test(image)

if __name__ == "__main__":
    main(sys.argv[1:])