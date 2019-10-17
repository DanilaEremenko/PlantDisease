def get_stdin_answer(text):
    while True:
        res_text = input("%s[y/n]" % text)
        if res_text == 'y':
            return True
        elif res_text == 'n':
            return False


def get_input_int(title=None, min=None, max=None):
    while True:
        if title != None:
            print(title)
        res = input()
        if res.isdigit():
            res = int(res)

            if (min != None) and (max != None):
                if (res >= min and res <= max):
                    return res
                else:
                    print("%d isn't between [%d,%d]" % (res, min, max))
            else:
                return res

        else:
            print("%s not an integer\n" % res)