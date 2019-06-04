import time

def section_logger(level=0):
    initial_t = time.time()
    sections = ['*', '->', '-', '·']
    paragraph = "\t"*level + sections[level]

    def section(str):
        print('\n{} {}. {} s'.format(paragraph, str, round(time.time() - initial_t)))

    return section

