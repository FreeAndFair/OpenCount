from PIL import Image, ImageDraw
import os
import subprocess
import sys
import random

try:
    from collections import Counter
except ImportError as e:
    from util import Counter
import multiprocessing as mp
import cPickle as pickle
import itertools
from grouping.partask import do_partask
from vendors import Vendor
import util

import ffwx
from ffwx.boxes import TargetBox, ContestBox, compute_box_ids

black = 200

do_save = True
do_test = True
export = False

flipped = {}


def num2pil(img):
    pilimg = Image.new("L", (len(img[0]), len(img)))
    pilimg.putdata([item for sublist in img for item in sublist])
    return pilimg


def load_pil(path):
    pilimg = Image.open(path)
    pilimg = pilimg.convert("L")
    # print 'loading', path
    # print 'isflipped', flipped
    if flipped != {} and flipped[path]:
        pilimg = pilimg.transpose(Image.ROTATE_180)
    return pilimg


def load_num(path="", pilimg=None):
    if pilimg is None:
        pilimg = load_pil(path)
    width, height = pilimg.size
    data = list(pilimg.getdata())
    data = [data[x:x + width] for x in range(0, width * height, width)]
    # print width, height
    return data


def load_threshold(image):

    def dorem(dat, block, boxes, replacewith=False):
        remove = []
        for x, y in boxes:
            if (x, y - block) in boxes and (x, y + block) in boxes:
                if (x - block, y) in boxes and (x + block, y) in boxes:
                    remove.append((x, y))
        for x, y in remove:
            for dy in range(block):
                for dx in range(block):
                    dat[y + dy][x + dx] = replacewith

    dat = load_num(image)
    block = 40
    boxes = {}
    for y in range(0, len(dat) - block, block):
        for x in range(0, len(dat[y]) - block, block):
            if sum(dat[y + dy][x + dx] < 240 for dy in range(0, block, 4) for dx in range(0, block, 4)) > block / 4 * block / 4 * 3 / 10:
                lst = [dat[y + dy][x + dx] <
                       240 for dy in range(0, block) for dx in range(0, block)]
                if sum(lst) > block * block * 7 / 10:
                    boxes[x, y] = True
    dorem(dat, block, boxes, replacewith=255)

    dat = [[x < black for x in y] for y in dat]
    block = 10
    boxes = {}
    for y in range(0, len(dat) - block, block):
        for x in range(0, len(dat[y]) - block, block):
            if sum(dat[y + dy][x + dx] for dy in range(0, block, 2) for dx in range(0, block, 2)) > block / 2 * block / 2 * 5 / 10:
                filled = sum(dat[y + dy][x + dx] for dy in range(block)
                             for dx in range(block)) > block * block * 9 / 10
                if filled:
                    boxes[x, y] = True

    dorem(dat, block, boxes, replacewith=255)

    dat = [[0 if x else 255 for x in y] for y in dat]
    if do_save and False:
        # TODO: This assumes that the dir 'tmp/' exists.
        load_pil(image).save(tmp + "/%s-a.png" % image.split("/")[1])
        num2pil(dat).save(tmp + "/%s-b.png" % image.split("/")[1])
    return dat


def find_lines(data):
    """
    Find all the lines we can on the image.

    For each pixel, if it is black:
        (1) Extend up and down as far as possible. Call those
              pixels part of one line.
        (2) Extend left and right as far as possible. Call those
              pixels part of one line.
        (3) Mark the lined pixels as 'used'.

    When extending, first move to the middle of the line. Then,
    do a DFS search going in the direction of the line, but only
    in the given direction.

    In order to improve efficiency, only test the black pixels on a grid.
    """
    height, width = len(data), len(data[0])

    def extend_ud(point):
        y, x = point
        while 0 < y and data[y][x] < black:
            y -= 1
        upper_y = y
        y, x = point
        low = len(data)
        while y < low and data[y][x] < black:
            y += 1
        lower_y = y
        return upper_y, lower_y

    def extend_lr(point):
        y, x = point
        while 0 < x and data[y][x] < black:
            x -= 1
        left_x = x
        y, x = point
        right = len(data[y])
        while x < right and data[y][x] < black:
            x += 1
        right_x = x
        return left_x, right_x

    def full_extend_ud(point):
        y, x = point
        l, r = extend_lr((y, x))
        x = (l + r) / 2 if l + r < 20 else x
        u1, d1 = extend_ud((y, x))
        return u1, d1

    LST = []

    def full_extend_lr_2(point):
        u, d = extend_ud(point)
        if d - u < 20:
            y = (u + d) / 2
        else:
            y = point[0]
        point = (y, point[1])

        lower = max(y - 10, 0)
        upper = min(y + 10, height)

        seen = []

        q = [point[0]]
        x = point[1] - 1
        while q and x > 0:
            q = list(set([dy + y for y in q for dy in [-1, 0, 1]
                          if lower <= dy + y < upper and data[dy + y][x] < black]))
            seen.extend(q)
            # LST.extend([(x,y) for y in q])
            x -= 1
        l = x
        q = [point[0]]
        x = point[1] + 1
        while q and x < width:
            q = list(set([dy + y for y in q for dy in [-1, 0, 1]
                          if lower <= dy + y < upper and data[dy + y][x] < black]))
            seen.extend(q)
            # LST.extend([(x,y) for y in q])
            x += 1
        r = x
        yy = sum(seen) / len(seen) if len(seen) else point[0]
        return yy, (l, r)

    def full_extend_ud_2(point):
        l, r = extend_lr(point)
        if r - l < 20:
            x = (l + r) / 2
        else:
            x = point[1]
        point = (point[0], x)

        lower = max(x - 10, 0)
        upper = min(x + 10, width)

        seen = []

        q = [point[1]]
        y = point[0] - 1
        while q and y > 0:
            q = list(set([dx + x for x in q for dx in [-1, 0, 1]
                          if lower <= dx + x < upper and data[y][dx + x] < black]))
            seen.extend(q)
            # LST.extend([(x,y) for y in q])
            y -= 1
        u = y
        q = [point[1]]
        y = point[0] + 1
        while q and y < height:
            q = list(set([dx + x for x in q for dx in [-1, 0, 1]
                          if lower <= dx + x < upper and data[y][dx + x] < black]))
            seen.extend(q)
            # LST.extend([(x,y) for y in q])
            y += 1
        d = y
        yy = sum(seen) / len(seen) if len(seen) else point[1]
        return yy, (u, d)

    foundy = {}
    foundx = {}
    lines = []
    YSKIP = 15
    XSKIP = 40
    for y in range(0, height, 1):
        for x in range(0, width, 1) if y % YSKIP == 0 else range(0, width, XSKIP):
            if not data[y][x] < black:
                # data[y][x] = black
                continue
            if y % YSKIP == 0 and (y / 3, x / 3) not in foundy:
                u, d = full_extend_ud((y, x))
                if d - u > 30:
                    xx, (u, d) = full_extend_ud_2((y, x))
                    if d - u > 30:
                        for dx in range(-10, 10, 3):
                            for q in range(u, d):
                                foundy[q / 3, (xx + dx) / 3] = True
                        lines.append(("V", (xx - 3, u, xx + 3, d)))

            if x % XSKIP == 0 and (y / 3, x / 3) not in foundx:
                # print 'h', newy, y, x
                yy, (l, r) = full_extend_lr_2((y, x))
                if r - l > 30:
                    for dy in range(-10, 10, 3):
                        for q in range(l, r):
                            foundx[(yy + dy) / 3, q / 3] = True
                    # print 'line starting from', x, y, data[y][x]
                    # LST.append((x-3,y-3,x+3,y+3))
                    # LST.append((l,y,r,y))
                    lines.append(("H", (l, yy - 3, r, yy + 3)))

    if do_save and False:
        num2pil(data).save(tmp + "/it.png")
        ct = Counter(LST)
        im = Image.new("RGB", (width, height), (255, 255, 255))
        d = ImageDraw.Draw(im)
        LST = list(set(LST))
        for each in LST:
            if len(each) == 4:
                d.rectangle(each, fill=(0, 0, 0))
            else:
                d.point(each, fill=(ct[each], 0, 0))
        im.save(tmp + "/asdf.png")

        im = Image.new("L", (width, height), 255)
        d = ImageDraw.Draw(im)
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] < black:
                    d.point((j, i), fill=0)
        im.save(tmp + "/asdf2.png")

    # print len(lines)
    # print lines
    return lines


def intersect(line1, line2):
    """
    Compute the intersection of two bounding boxes.
    """
    top = max(line1[1], line2[1])
    bottom = min(line1[3], line2[3])
    left = max(line1[0], line2[0])
    right = min(line1[2], line2[2])
    if bottom > top and right > left:
        return left, top, right, bottom
    else:
        return None


def union(line1, line2):
    """
    Compute the union of two bounding boxes.
    """
    top = min(line1[1], line2[1])
    bottom = max(line1[3], line2[3])
    left = min(line1[0], line2[0])
    right = max(line1[2], line2[2])
    return left, top, right, bottom


def dfs(graph, start):
    """
    Run a DFS search on a graph starting at a given vertex.
    Return the list of seen graphs.
    """
    s = [start]
    seen = {}
    while s != []:
        top = s.pop()
        if top in seen:
            continue
        seen[top] = True
        s += graph[top]
    return seen.keys()


@util.pdb_on_crash
def extend_to_line(lines, width, height):
    table = [[False] * (width + 200) for _ in range(height + 200)]
    for d, each in lines:
        if d == 'V':
            (l, u, r, d) = each
            for x in range(l, r):
                for y in range(u, d):
                    table[y][x] = True

    print "RECTANGLE FIX", len(lines)

    new = []
    for direc, each in lines:
        if direc == 'H':
            l, u, r, d = each
            if any(table[u][x] for x in range((l + 9 * r) / 10, r)):
                new.append((direc, each))
            else:
                pos = [x for x in range(
                    (l + 9 * r) / 10, min(r + (r - l) / 2, width)) if table[u][x]]
                if len(pos):
                    new.append((direc, (l, u, pos[0] + (pos[0] - l) / 10, d)))
                else:
                    new.append((direc, (l, u, r, d)))
        else:
            new.append((direc, each))

    print len(new)

    return new


def to_graph(lines, width, height, minsize, giventargets):
    """
    Convert a set of lines to graph where lines are vertexes, and
    there is an edge between two lines when they intersect. This
    prepares for finding all rectangles.

    First, we need to find lines which were accidentally split in to
    pieces, and merge them together. That is, if there are two
    horizontal lines which intersect, or two vertical lines which
    intersect, then merge them in to one line of the union of the two.

    We do this by making another graph of all horizontal lines and
    adding edges between the lines when they touch. Then we run a
    connnected components algorithm over it and take the union.
    This is done by creating an width*height array, storing where all
    of the lines are, and finding when they touch.

    Then it is easy to do an all-pairs algorithm to find the intersecting
    vertical and horizontal lines.
    """
    print width, height

    # Extend the lines.
    def extend(line):
        if line[0] == 'V':
            l, u, r, d = line[1]
            ext = int(round((d - u) * 0.02))
            return ('V', (l, u - ext if u - ext >= 0 else 0, r, d + ext if d + ext < height else height - 1))
        if line[0] == 'H':
            l, u, r, d = line[1]
            ext = int(round((r - l) * 0.02))
            return ('H', (l - ext if l - ext >= 0 else 0, u, r + ext if r + ext < width else width - 1, d))

    for _ in range(2):
        print "THERE ARE", len(lines)
        lines = map(extend, lines)

        for direction in ['H', 'V']:
            table = [[None] * (width + 200) for _ in range(height + 200)]
            equal = []
            for full in lines:
                if full[0] != direction:
                    continue
                _, (l, u, r, d) = full
                for x in range(l, r):
                    for y in range(u, d):
                        if table[y][x] is not None:
                            equal.append((table[y][x], full))
                        else:
                            table[y][x] = full
            equal = list(set(equal))
            # print equal
            graph = {}
            for v1, v2 in equal:
                if v1 not in graph:
                    graph[v1] = []
                if v2 not in graph:
                    graph[v2] = []
                graph[v1].append(v2)
                graph[v2].append(v1)
            # print graph
            seen = {}
            new = []
            for el in graph.keys():
                if el in seen:
                    continue
                makeequal = dfs(graph, el)
                for each in makeequal:
                    seen[each] = True
                new.append((makeequal[0][0], reduce(
                    union, [x[1] for x in makeequal])))
            for line in lines:
                if line not in seen:
                    new.append(line)
            lines = new
    print "THERE ARE END", len(lines)
    print list(sorted([area(x[1]) for x in lines]))
    print minsize
    lines = [x for x in lines if x[1][2] - x[1][0] >
             width / 10 or x[1][3] - x[1][1] > height / 30]
    print "THERE ARE END", len(lines)

    lines = extend_to_line(lines, width, height)

    new_lines = []
    for k, line in lines:
        if k == 'H':
            if any(line[0] <= (x[0] + x[2]) / 2 <= line[2] for x in giventargets):
                new_lines.append((k, line))
        if k == 'V':
            if any(line[1] <= (x[1] + x[3]) / 2 <= line[3] for x in giventargets):
                new_lines.append((k, line))
    lines = new_lines

    vertexes = dict((x, []) for _, x in lines)

    boxes = []
    for way1, line1 in lines:
        for way2, line2 in lines:
            if way1 != way2:
                if intersect(line1, line2):
                    boxes.append(intersect(line1, line2))
                    vertexes[line1].append(line2)
    print 'finished', len(str(vertexes)), len(boxes)
    return boxes, dict((k, v) for k, v in vertexes.items() if v != [])


def find_squares(graph, minarea):
    """
    Given a graph (vertexes are lines, edges when they intersect),
    return the squares that are in the graph.
    A square is when the DFS finds a back-edge where the difference in
    the preorders of the two nodes is 4.
    """
    def dfs_square(stack, debug=False):
        if debug:
            print ".  " * len(stack), stack[-1]
        if len(stack) == 4:
            if stack[0] in graph[stack[-1]]:
                tores = intersect(
                    union(stack[0], stack[2]), union(stack[1], stack[3]))
                if area(tores) > minarea:
                    return [tores]
            return [None]
        res = []
        for vertex in graph[stack[-1]]:
            if vertex in stack:
                continue
            res += dfs_square(stack + [vertex], debug)
        return res

    # result = [dfs_square([start]) for start in graph]
    # result = [x for sublist in result for x in sublist]
    # return list(set([x for x in result if x]))
    result = {}
    for i, start in enumerate(graph):
        # print 'on', i, 'of', len(graph)
        for each in dfs_square([start]):
            if each:
                result[each] = True
    return result.keys()


def area(x):
    if x is None:
        return 0
    return (x[2] - x[0]) * (x[3] - x[1])


def do_extract(name, img, squares, giventargets):
    """
    Find all contests and extract them.

    Start with the smallest sized bounding box, and check if it
    contains any voting targets.

    If it does, it's a contest. Remove it and the targets that
    are inside of it. Move on to next biggest.

    If not, then remove it and move on to the next biggest.

    For each contest, see if there are any previously-identified
    voting targets that were removed by a smaller bounding box,
    if there are, warn the operator. This means one contest
    encloses another.
    """

    targets = [x for x in giventargets]
    avg_targ_area = sum(map(area, targets)) / len(targets)
    squares = [x for x in squares if area(x) > avg_targ_area * 2]  # xxx

    contests = []

    # print "T", targets
    for sq in sorted(squares, key=area):
        if sq in targets:
            continue
        inside = [t for t in targets if area(
            intersect(sq, t)) > area(t) / 2]  # xxx
        # print sq
        if inside != [] and sq[3] - sq[1] > 2 * (t[3] - t[1]):
            # print "Adding a contest", sq, inside, [area(intersect(sq, t)) for
            # t in inside]
            contests.append(sq)
            targets = [x for x in targets if x not in inside]

    if targets != []:
        print "Was left with", targets
    keepgoing = True
    while keepgoing:
        keepgoing = False
        for target in giventargets:
            # print 'this target', target
            tomerge = [x for x in contests if intersect(
                x, target) == target]  # xxx
            if len(tomerge) > 1:
                # Find the smallest subset to merge which overlap all targets
                # in all contests.
                maxcontest = None
                must_include_targets = sum(
                    [[x for x in giventargets if intersect(c, x)] for c in tomerge], [])  # xxx
                print 'must include', must_include_targets
                found = False
                for group_size in range(1, len(tomerge) + 1):
                    if found:
                        break
                    for comb in itertools.combinations(tomerge, group_size):
                        thiscontest = reduce(union, comb)
                        print 'this', thiscontest
                        print 'for each', [intersect(targ, thiscontest) for targ in must_include_targets]
                        if all(intersect(targ, thiscontest) for targ in must_include_targets):
                            print 'yes'
                            maxcontest = thiscontest
                            found = True
                            break
                print "MERGING", tomerge
                contests = [
                    x for x in contests if x not in tomerge] + [maxcontest]  # xxx
                keepgoing = True
                break
            elif len(tomerge) < 1:
                print "Target", target, "Not in any contest on ballot", name

    def samecolumn(a, b):
        if (abs(a[0] - b[0]) < 30 or abs(a[2] - b[2]) < 30):
            if abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) < 100:
                return True
        return False

    def height(x):
        if x is None:
            return 0
        return x[3] - x[1]

    if len(contests) > 2 * len(giventargets) / 3:
        equivs = list(contests)
        prev = 0
        h = giventargets[0][3] - giventargets[0][1]
        print "start", contests, h
        while prev != len(equivs):
            prev = len(equivs)
            new = []
            skip = {}
            for a in equivs:
                if a in skip:
                    continue
                print "On ", a
                if a[3] - a[1] > len([x for x in giventargets if intersect(x, a)]) * h * 5:
                    print 'abort 1', len([x for x in giventargets if intersect(x, a)]), a[3] - a[1]
                    new.append(a)
                    continue
                found = None
                for b in equivs:
                    if b[3] - b[1] > len([x for x in giventargets if intersect(x, b)]) * h * 5:
                        print 'abort 2', len([x for x in giventargets if intersect(x, b)]), b[3] - b[1]
                        continue
                    if a == b:
                        continue
                    if abs(b[3] - a[1]) < 30 and samecolumn(a, b):
                        print 'case 2', b
                        found = b
                        break
                if found is not None:
                    print 'merge'
                    new.append(union(a, found))
                    skip[found] = True
                    skip[a] = True
                else:
                    new.append(a)
            equivs = new

            # Find which contests we've detected as equal,
            # and hopefully not ones which happen to overlap a bit,
            # and merge them together.
            equivs = []
            while new:
                s = [x for x in new if height(
                    intersect(x, new[0])) >= h and samecolumn(x, new[0])]
                equivs.append(reduce(union, s))
                new = [x for x in new if x not in s]

            # print "RES", len(equivs), equivs

        contests = equivs

    # print "C", contests
    for cont in contests:
        if export:
            im = img.crop(cont)
            cname = tmp + "/" + str(sum(im.histogram()[:100])) + ".png"
            im = img.crop(cont)
            im.save(cname)

    if do_save or do_test:
        new = Image.new("RGB", img.size, (255, 255, 255))
        imd = ImageDraw.Draw(new)
        for box in contests:
            c = (int(random.random() * 200), int(random.random()
                                                 * 155 + 100), int(random.random() * 155 + 100))
            imd.rectangle(box, fill=c)
        # print "GIVEN", giventargets
        for box in giventargets:
            # print box, area(box)
            imd.rectangle(box, fill=(255, 0, 0))
        new.save(tmp + "/" + name + "-fullboxed.png")

    return contests


def extract_contest(args):
    if len(args) == 3:
        image_path, giventargets, queue = args
        returnimage = True
    elif len(args) == 4:
        image_path, giventargets, returnimage, queue = args
    else:
        raise Error("Wrong number of args")

    print len(giventargets), giventargets

    print "processing", image_path
    data = load_threshold(image_path)
    # data = load_num(image_path)
    print 'loaded'
    lines = find_lines(data)
    lines += [('V', (len(data[0]) - 20, 0, len(data[0]), len(data)))]
    # print "calling with args", lines, len(data[0]), len(data),
    # max(giventargets[0][2]-giventargets[0][0],giventargets[0][3]-giventargets[0][1])
    boxes, graph = to_graph(lines, len(data[0]), len(
        data), area(giventargets[0])**.5, giventargets)
    print 'tograph'
    squares = find_squares(graph, area(giventargets[0]))
    print 'findsquares'
    squares = sorted(squares, key=lambda x: -(x[2] - x[0]) * (x[3] - x[1]))
    # print lines
    # print squares

    filename = ".".join(image_path.split("/")[-2:])[:-4]
    if do_save:
        show = num2pil(data)
        # load_pil(image_path).copy().convert("RGB")#
        new = Image.new("RGB", show.size, (255, 255, 255))
        imd = ImageDraw.Draw(new)
        for line in [x[1] for x in lines]:
            imd.rectangle(line, outline=(0, 0, 0))
        for line in boxes:
            imd.rectangle(line, fill=(0, 0, 255))

        print len(squares), "NUM"

        new.save(tmp + "/" + filename + "-line.png")

        new = Image.new("RGB", show.size, (255, 255, 255))
        imd = ImageDraw.Draw(new)
        for line in graph:
            imd.rectangle(line, outline=(0, 0, 0))
        for line in boxes:
            imd.rectangle(line, fill=(0, 0, 255))

        print len(squares), "NUM"

        new.save(tmp + "/" + filename + "-line-2.png")

        for l, u, r, d in squares:
            c = (int(random.random() * 255),
                 int(random.random() * 255), int(random.random() * 255))
            imd.rectangle((l, u, r, d), fill=c)
        new.save(tmp + "/" + filename + "-box.png")

    if do_save or export or do_test:
        loadedimage = load_pil(image_path)
    else:
        loadedimage = None

    # print "GET ARG", image_path, image_path.split("/")[-1]

    print len(giventargets), giventargets

    final = do_extract(filename,
                       loadedimage, squares, giventargets)

    # print "before"
    # print final
    # final = remove_contest_overlap(final, giventargets)
    # print "after"
    # print final

    # os.popen("open tmp/*")
    # exit(0)

    queue.put(1)

    if returnimage:
        return data, final
    else:
        return final


def ballot_preprocess(i, f, image, contests, targets, lang, vendor):
    """
    Preprocess a ballot and turn it in to its corresponding data.
    For each contest, record the ballot ID, the contest bounding box,
    as well as the text associated with the contest.
    """
    sub = os.path.join(tmp + "", f.split("/")[-1].split(".")[0] + "-dir")
    # print "SUB IS", sub
    if not os.path.exists(sub):
        os.mkdir(sub)
    res = []
    print "CONTESTS", contests

    all_boxes = []
    for l, u, r, d in contests:
        all_boxes.append(ContestBox(l, u, r, d))

    for l, u, r, d in targets:
        all_boxes.append(TargetBox(l, u, r, d))

    assocs, _ = compute_box_ids(all_boxes)

    def grab(box):
        return (box.x1, box.y1, box.x2, box.y2)

    for c, targets in assocs.values():
        c = grab(c)
        targets = map(grab, targets)
        print "TOMAKE", c
        if not os.path.exists(os.path.join(sub, "-".join(map(str, c)))):
            os.mkdir(os.path.join(sub, "-".join(map(str, c))))
        t = compare_preprocess(lang, os.path.join(sub, "-".join(map(str, c))),
                               image, c, targets, vendor)
        res.append((i, c, t))
    print "RESULTING", res
    return res


def compare_preprocess(lang, path, image, contest, targets, vendor):
    """
    Identifies the text associated with the contest.

    Split the contest in to "stripes", one for each voting target,
    and one for the title. OCR the text and record it.
    """

    # targets = [x for x in targets if area(intersect(contest, x))] # xxx
    cont_area = None

    print path
    print 'targs', len(targets), targets

    if vendor:
        boxes = vendor.split_contest_to_targets(image, contest, targets)
    else:
        boxes = Vendor.Vendor(None).split_contest_to_targets(
            image, contest, targets)

    l, u, r, d = contest
    blocks = []
    print 'lenbox', len(boxes), boxes
    for count, (upper, lower) in boxes:
        istarget = (count != 0)
        print upper, lower
        if upper == lower:
            blocks.append((istarget, ""))
            continue
        name = os.path.join(path, str(count) + ".tif")
        if os.path.exists(name + ".txt"):
            txt = open(name + ".txt").read().decode('utf8')
            if txt != '':
                print 'Found'
                blocks.append((istarget, txt))
                continue
            else:
                print 'Empty'
        # print "POS", upper, lower
        # print len(cont_area[upper:lower])
        if not os.path.exists(name):
            if cont_area is None:
                cont_area = load_num(pilimg=num2pil(image).crop(
                    (l + 10, u + 10, r - 10, d - 10)))
            img = num2pil(cont_area[upper:lower])
            img.save(name)

        try:
            subprocess.check_call(
                ("tesseract-ocr", name, name, "-l", lang))
        except subprocess.CalledProcessError:
            raise ffwx.Panel.FatalError(
                'Error invoking tesseract')
        with open(name + '.txt') as f:
            txt = f.read().decode('utf8')

        if os.path.exists(name + ".txt"):
            blocks.append((istarget, txt))
        else:
            print "-" * 40
            print "OCR FAILED"
            print name
            print path
            print contest
            print lang
            print count, upper, lower
            print "-" * 40
            blocks.append((istarget, ""))

    print 'retlen', len(blocks)
    return blocks

# import editdist
try:
    from Levenshtein import distance
except:
    print "Error: Edit Distance module not loaded"
    if __name__ != '__main__':
        print "Exiting"
        exit(1)


def row_dist(a, b):
    if type(a) == unicode or type(b) == unicode:
        return distance(unicode(a), unicode(b))
    else:
        return distance(a, b)
    # v = editdist.distance(a.encode("ascii", "ignore"),
    #                      b.encode("ascii", "ignore"))
    # print 'r', v, a == b
    return v
    """
    Compute the edit distance between two strings.
    """
    if a == b:
        return 0
    prev = None
    curr = range(len(b) + 1)

    for i in range(len(a)):
        # print curr
        prev = curr
        curr = [0] * (len(b) + 1)
        curr[0] = i + 1
        for j in range(len(b)):
            curr[j + 1] = min(prev[j + 1] + 1,
                              curr[j] + 1,
                              prev[j] + (a[i] != b[j]))
    return curr[-1]


count = 0


def compare(otexts1, otexts2, debug=False):
    """
    Compute the distance between two contests.

    This distance allows the order of the targets to shuffle,
    but does not allow the text within a target to reorder.

    We do this with an approximation of the weighted disjoint
    set cover problem. We sort the edges by weight, and take the
    minimum weight one, remove the corresponding nodes, and recurse.

    The distance is sum of the edges picked, as well as the sum
    of the unused vertexes.
    """

    if len(otexts1) != len(otexts2):
        print "Possible error: tried to compare distance of two contests with different number of targets."
        return 1 << 30

    def fixup(s):
        words = s.split()
        found = 0
        for item, sep in [('Party Preference: Republican', 3), ('Party Preference: Democratic', 3),
                          ('MEMBER OF THE STATE ASSEMBLY', 5)]:
            for i in range(len(words) - (sep - 1)):
                combined = " ".join(words[i:i + sep])
                if abs(len(combined) - len(item)) < len(item) / 10:
                    if row_dist(combined, item) < len(item) / 5:
                        words[i:i + sep] = item.split(" ")
                        found += len(item)
        return " ".join(words), found

    texts1, founds1 = zip(*[fixup(x) for t, x in otexts1 if t])
    texts2, founds2 = zip(*[fixup(x) for t, x in otexts2 if t])
    # Text associated with targets only
    ordering1 = range(len(texts1))
    ordering2 = range(len(texts2))
    size = sum(map(len, [x for _, x in otexts1])) + sum(map(len,
                                                            [x for _, x in otexts2])) - sum(founds1) - sum(founds2)
    # print 'size', size
    if size == 0:
        print "Possible Error: A contest has no text associated with it"
        return [(1 << 30, (len(texts1), 0, 0)) for _ in range(len(texts1))], (1 << 30, 0)

    titles1 = [x for t, x in otexts1 if not t]
    titles2 = [x for t, x in otexts2 if not t]
    val = sum(row_dist(*x) for x in zip(titles1, titles2))
    # print 'dist of titles is', val

    all_vals = []
    for num_writeins in [0]:  # range(len(texts2)):
        rottexts2 = [[texts2[i] for _, i in get_order(
            len(texts2), order, num_writeins)] for order in range(len(texts2))]
        values = [(sum(row_dist(a, b) for a, b in zip(texts1, t2)), i)
                  for i, t2 in enumerate(rottexts2)]
        if debug:
            print "DEBUG", size, size - sum(map(len, titles1)) - sum(map(len, titles2))
            print num_writeins
            print [([row_dist(a, b) for a, b in zip(texts1, t2)], i) for i, t2 in enumerate(rottexts2)]
            print map(len, texts1), map(len, texts2)
            print min(values)
        # print values
        minweight, order = min(values)

        # print 'min', order, minweight

        all_vals.append((minweight, order, num_writeins))
    # print "BEST:", best_val
    # print 'so should be equal'
    # print texts1
    # print
    # texts2[best_val[1]:-best_val[2]]+texts2[:best_val[1]]+texts2[-best_val[2]:]
    all_vals = sorted(all_vals)
    res = {}
    best = 1 << 30, None
    for weight, order, num_writeins in all_vals:
        if float(weight + val) / size < best[0]:
            best = float(weight + val) / size, num_writeins
        res[num_writeins] = (float(weight + val) / size,
                             (len(texts1), order, num_writeins))
    # print otexts1
    # print otexts2
    # print "res", [x[1] for x in sorted(res.items())], best
    return [x[1] for x in sorted(res.items())], best


def get_order(length, order, num_writeins):
    lst = range(length)
    if num_writeins == 0:
        new_order = lst[order:] + lst[:order]
    else:
        new_order = lst[order:-num_writeins] + \
            lst[:order] + lst[-num_writeins:]
    return list(zip(lst, new_order))


def first_pass(contests, languages):
    """
    Split a set of contests in to a set of sets, where each
    set contains the same number of voting targets of the same language.
    """
    ht = {}
    i = 0
    for each in contests:
        key = (len(each[2]), None if each[0]
               not in languages else languages[each[0]])
        if key not in ht:
            ht[key] = []
        ht[key].append(each)
    return [x for x in sorted(ht.items())]


class Contest:

    def __init__(self, contests_text, cid, const=.2):
        self.contests_text = contests_text
        self.cid = cid
        self.const = const
        # CID -> [(distance, order, numwritein)]
        self.similarity = {}
        self.parent = self
        self.depth = 0
        self.children = []
        self.writein_num = 0

    def all_children(self):
        res = [self]
        for child in self.children:
            res += child.all_children()
        return res

    def get_root(self):
        while self.parent != self.parent.parent:
            self.parent = self.parent.parent
        return self.parent

    def is_close(self, other, num_writein):
        group1 = self.all_children()
        group2 = other.all_children()
        best = 1 << 31, None
        # print 'joining', len(group1), len(group2)
        for nwi in set([self.writein_num, other.writein_num, num_writein]):
            distance = 0
            for c1 in group1:
                for c2 in group2:
                    if c2.cid not in c1.similarity:
                        distance += 1
                    else:
                        distance += c1.similarity[c2.cid][nwi][0]
            distance /= len(group1) * len(group2)
            # print nwi, distance
            if distance < best[0]:
                best = distance, nwi
        # print 'pick', best
        return best[0] < self.const, best[1]

    def join(self, new_parent, num_writein):
        if self.get_root() == new_parent.get_root():
            return

        root1 = self.parent
        root2 = new_parent.parent

        close, winum = root1.is_close(root2, num_writein)
        if not close:
            return

        if root1.depth < root2.depth:
            root1.parent = root2
            root2.children.append(root1)
            root2.writein_num = winum
        elif root2.depth < root1.depth:
            root2.parent = root1
            root1.children.append(root1)
            root1.writein_num = winum
        else:
            root1.parent = root2
            root1.depth += 1
            root2.children.append(root1)
            root2.writein_num = winum


def do_group_pairing_map(args):
    global tmp

    args = args[0]
    # for k,v in list(zip(globals(), [type(v) for k,v in globals().items()])):
    #    print k,v

    items, contests_text = args
    # out = open(tmp+"/group_dump/"+str(items[0]), "w")
    x = 0
    for i in items:
        lst = []
        for j in range(len(contests_text)):
            if x % 10000 == 0:
                print x
            x += 1
            # print ((i,j),compare(contests_text[i][2], contests_text[j][2]))
            lst.append(
                ((i, j), compare(contests_text[i][2], contests_text[j][2])))
        # out.write("\n".join(map(str,lst))+"\n")
        pickle.dump(lst, open(tmp + "/group_dump/" + str(i), "w"))
    # out.close()
    return []


def group_by_pairing(contests_text, CONST):
    """
    Group contests together by pairing them one at a time.

    Currently this is very slow. It's going to run n^2 comparisons,
    and then do a linear scan through each of them to make the groups.
    """
    global tmp

    contests = [Contest(contests_text, i, CONST)
                for i in range(len(contests_text))]

    # args = [(i,cont1,j,cont2) for i,cont1 in enumerate(contests_text) for j,cont2 in enumerate(contests_text)]

    # """
    if not os.path.exists(tmp + "/group_dump"):
        os.mkdir(tmp + "/group_dump")
    else:
        os.popen("rm " + tmp + "/group_dump/*")

    """
    print "Prepare"
    pool = mp.Pool(mp.cpu_count())
    print "Start"
    data = [[] for _ in range(mp.cpu_count())]
    for i in range(len(contests_text)):
        data[i%len(data)].append(i)
    print "GO UP TO", (len(contests_text)**2)/mp.cpu_count()
    data = [(x, contests_text) for x in data]
    pool.map(do_group_pairing_map, data)
    pool.close()
    pool.join()
    # """
    if False:
        do_group_pairing_map([(range(len(contests_text)), contests_text)])
    else:
        num = mp.cpu_count()
        data = [[] for _ in range(num)]
        for i in range(len(contests_text)):
            data[i % len(data)].append(i)
        print "GO UP TO", (len(contests_text)**2) / num
        data = [(x, contests_text) for x in data]
        do_partask(do_group_pairing_map, data, N=num)

    diff = {}
    print len(contests_text)
    for i in range(len(contests_text)):
        if i % 100 == 0:
            print 'load', i
        d = pickle.load(open(tmp + "/group_dump/" + str(i)))
        for k, v in d:
            diff[k] = v

    print "Done"
    diff = sorted(diff.items(), key=lambda x: x[1][1][0])
    print len(diff)
    # print diff[0]

    for (k1, k2), (dmap, best) in diff:
        if k1 == k2:
            print 'eq', k1, k2, dmap, best
        contests[k1].similarity[k2] = dmap
    print "Created"
    for (k1, k2), (dmap, best) in diff:
        if best[0] > CONST:
            continue
        if k1 == k2:
            continue
        contests[k1].join(contests[k2], best[1])
        # print 'join', contests_text[k1][0], contests_text[k2][0]
        # print 'data', contests[k1].writein_num, contests[k2].writein_num
        # print contests_text[k1][2][1]
        # print contests_text[k2][2][1]
    print "Traverse"
    seen = {}
    res = []
    for contest in contests:
        # print 'try', contest.cid,
        contest = contest.get_root()
        if contest in seen:
            continue
        # print "SEE", contest
        # contest.dominating_set()
        seen[contest] = True
        v = [x.cid for x in contest.all_children()]
        # print "CHILDREN", v
        write = contest.writein_num
        # print "FOR THIS GROUP", write
        if contest.cid in contest.similarity:
            print 'case 1', get_order(*contest.similarity[contest.cid][write][1]), contest.similarity[contest.cid][write][1]
            this = [(contests_text[contest.cid][:2], get_order(
                *contest.similarity[contest.cid][write][1]))]
        else:
            # print 'case 2', range(len(contests_text[contest.cid][:2])-1)
            v = []
            l = range(len(contests_text[contest.cid][:2]) - 1)
            this = [(contests_text[contest.cid][:2], zip(l, l))]
        # print "Base"
        # print list(enumerate(contests_text[contest.cid][2][1:]))
        for x in v:
            if x == contest.cid:
                continue
            # print contest.similarity[x]
            # print contest.similarity[x][write][1], get_order(*contest.similarity[x][write][1])
            # print "This", list(enumerate(contests_text[x][2][1:]))
            this.append((contests_text[x][:2], get_order(
                *contest.similarity[x][write][1])))
        # print this
        print "ADD", len(res), this
        res.append(this)
    print "Workload reduction"
    print map(len, res)
    print len([x for x in map(len, res) if x > 3]) + sum([x for x in map(len, res) if x <= 3]), sum(map(len, res))
    print "Factor", float(len([x for x in map(len, res) if x > 3]) + sum([x for x in map(len, res) if x <= 3])) / sum(map(len, res))
    return res


def full_group(contests_text, key):
    print "Linear Scan"

    if key[1] == 'eng':
        CONST = .2
    elif key[1] == 'spa':
        CONST = .2
    elif key[1] == 'vie':
        CONST = .25
    elif key[1] == 'kor':
        CONST = .3
    elif key[1] == 'chi_sim':
        CONST = .3
    else:
        CONST = .2

    debug = []

    contests_text = sorted(
        contests_text, key=lambda x: sum(len(v[1]) for v in x[2]))
    joins = dict((i, []) for i in range(len(contests_text)))
    for offset in range(1, 2):
        for i, (c1, c2) in enumerate(zip(contests_text, contests_text[offset:])):
            data, (score, winum) = compare(c1[2], c2[2])
            debug.append((score, (c1[2][0], c2[2][0])))
            if score < CONST / 2:
                # print 'merged', c1[2], c2[2]
                joins[i].append(i + offset)
                joins[i + offset].append(i)

    seen = {}
    exclude = {}
    for i in joins:
        if i in seen:
            continue
        items = dfs(joins, i)
        first = min(items)
        for each in items:
            seen[each] = True
        for each in items:
            if first != each:
                exclude[each] = first

    # print sorted(exclude.items())

    new_indexs = [x for x in range(len(contests_text)) if x not in exclude]
    new_contests = [contests_text[x] for x in new_indexs]

    print "Of sizes", len(contests_text), len(new_contests)
    # for x in new_contests[::100]:
    #    print x
    newgroups = []
    STEP = 1000
    print "Splitting to smaller subproblems:", round(.5 + (float(len(new_contests)) / STEP))
    for iternum in range(0, len(new_contests), STEP):
        print "SUBPROB", iternum / STEP
        newgroups += group_by_pairing(
            new_contests[iternum:min(iternum + STEP, len(new_contests))], CONST)

    mapping = {}
    for i, each in enumerate(newgroups):
        for item in each:
            mapping[item[0][0], tuple(item[0][1])] = i
    # print "mapping", mapping

    for dst, src in exclude.items():
        # print "Get", dst, "from", src
        bid, cids = contests_text[src][:2]
        index = mapping[bid, tuple(cids)]
        find = newgroups[index][0][0]
        text = [text for bid, cid, text in contests_text if (bid, cid) == find][
            0]
        data, (score, winum) = compare(text, contests_text[dst][2])
        newgroups[index].append(
            (contests_text[dst][:2], get_order(*data[winum][1])))

    print "SO GET"
    # print sorted(map(hash,map(str,map(sorted,groups))))
    print sorted(map(sorted, newgroups))

    return newgroups


def equ_class(contests, languages):
    # print "EQU", contests
    # print map(len, contests)
    # print contests
    contests = [x for sublist in contests for x in sublist]
    # print contests
    groups = first_pass(contests, languages)
    # Each group is known to be different.
    result = []
    print "Go up to", len(groups)
    for i, (key, group) in enumerate(groups):
        print "-" * 50
        print "ON GROUP", i, key, len(group)
        print "-" * 50
        result += full_group(group, key)
        print "Finished one group"
        print "Total length", len(result)

    # print "RETURNING", result
    return result


def get_target_to_contest(contests, targets):
    if 1 == 1:
        all_boxes = []
        for l, u, r, d in contests:
            all_boxes.append(ContestBox(l, u, r, d))

        for l, u, r, d in targets:
            all_boxes.append(TargetBox(l, u, r, d))

        assocs, _ = compute_box_ids(all_boxes)

        def grab(box): return (box.x1, box.y1, box.x2, box.y2)

        target_to_contest = {}
        for contest, all_targets in assocs.values():
            for target in all_targets:
                target_to_contest[grab(target)] = contests.index(grab(contest))
        return target_to_contest


def merge_contests(ballot_data, fulltargets):
    """
    Given a set of bounding boxes, merge together those which
    are different boundingboxes but are, in reality, the same contest.
    """
    # pdb.set_trace()
    new_data = []
    for ballot, targets in zip(ballot_data, fulltargets):
        # print 'next'
        new_ballot = []
        seen_so_far = []

        target_to_contest = get_target_to_contest(
            [x[1] for x in ballot], sum(targets, []))

        for group in targets:
            # for c,targets in assocs.values():

            # print 'targs is', group
            # indexs in ballot of contests which are equal
            # equal = [i for t in group for i,(_,bounding,_) in enumerate(ballot) if intersect(t, bounding) == t]
            equal = [target_to_contest[x] for x in group]
            equal_uniq = list(set(equal))
            if any(x in seen_so_far for x in equal_uniq) or equal_uniq == []:
                raise Exception()
            seen_so_far.extend(equal_uniq)
            # print equal_uniq
            merged = sum([ballot[x][2] for x in equal_uniq], [])
            new_ballot.append(
                (ballot[equal[0]][0], [ballot[x][1] for x in equal_uniq], merged))
        new_data.append(new_ballot)
    # print new_data
    return new_data


def do_extend(args):
    txt, c1, c2, t1, t2 = args
    data, (score, winum) = compare(txt, t1[2] + t2[2])
    if score < .2:
        # print "THEY ARE EQUAL"
        res = (c1, c2)
        # print 'txt', t1, t2
        newgroup = ((c1[0], [c1[1], c2[1]], t1[2] + t2[2]),
                    get_order(*data[winum][1]))
        return res, newgroup
    return None


def extend_multibox(ballots, box1, box2, orders):
    ballot = ballots[box1[0]]
    txt1 = [x for x in ballot if x[:2] == box1][0]
    txt2 = [x for x in ballot if x[:2] == box2][0]
    txt = txt1[2] + txt2[2]
    res = []
    newgroup = []

    tocompare = []
    for bid, order in enumerate(orders):
        # print 'BID IS', bid
        for c1, c2 in order:
            t1 = [x for x in ballots[bid] if x[:2] == c1][0]
            t2 = [x for x in ballots[bid] if x[:2] == c2][0]
            if len(t1[2]) + len(t2[2]) != len(txt1[2]) + len(txt2[2]):
                continue
            tocompare.append((txt, c1, c2, t1, t2))
    pool = mp.Pool(mp.cpu_count())
    res = pool.map(do_extend, tocompare)
    pool.close()
    pool.join()
    print "RESULT", res
    res = [x for x in res if x is not None]
    res, newgroup = zip(*res)
    print "RESULT", res
    print "NEWGROUP", newgroup

    return res, newgroup


@util.pdb_on_crash
def find_contests(t, paths, giventargets):
    """
    Input:
        str T:
        list PATHS:
        list GIVENTARGETS: G[i][j][k] := k-th target of j-th contest of i-th ballot.
    """
    global tmp
    # print "ARGS", (t, paths, giventargets)
    # exit(0)
    # paths = paths[80:400]
    # giventargets = giventargets[80:400]

    """
    giventargets += giventargets
    giventargets += giventargets
    npaths = list(paths)
    npaths += [x.replace("bal_0", "bal_1") if os.path.exists(x.replace("bal_0", "bal_1")) else x for x in paths]
    npaths += [x.replace("bal_0", "bal_2") if os.path.exists(x.replace("bal_0", "bal_2")) else x for x in paths]
    npaths += [x.replace("bal_0", "bal_3") if os.path.exists(x.replace("bal_0", "bal_3")) else x for x in paths]
    paths = npaths
    """
    if t[-1] != '/':
        t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    os.popen("rm -r " + tmp.replace(" ", "\\ ") + "*")

    manager = mp.Manager()
    queue = manager.Queue()
    args = [(f, sum(giventargets[i], []), False, queue)
            for i, f in enumerate(paths)]
    # args = [x for x in args if '0/bal_0_side_1' in x[0]]
    pool = mp.Pool(mp.cpu_count())
    res = [None]

    def done(x): res[0] = x
    pool.map_async(extract_contest, args, callback=done)
    got = 0
    while got < len(args):
        sys.stderr.write('.')
        val = queue.get(block=True)
        util.Gauges.infer_contests.tick()
        got += 1

    pool.close()
    pool.join()
    return res[0]
    # print "RETURNING", ballots
    reverse = sorted(enumerate(paths), key=lambda x: x[1])
    for i in range(0, len(reverse), 4):
        print reverse[i:i + 4]
        full = [ballots[x] for x, _ in reverse[i:i + 4]]
        if full == []:
            continue
        order = []
        for each in full:
            order.append(sorted(each, key=lambda x: (x[0] / 200, x[1])))
        print 'i get', i, order
        for cs in zip(*order):
            err = 0
            for coords in zip(*cs):
                avg = sum(coords) / len(coords)
                err += sum(abs(x - avg) for x in coords)
            if err > 100:
                print 'err', err, paths[reverse[i][0]], reverse[i]
                print cs
    return ballots


def group_given_contests_map(arg):
    vendor, lang_map, giventargets, (i, (f, conts)) = arg
    print f
    im = load_num(f)
    lang = lang_map[f] if f in lang_map else 'eng'
    return ballot_preprocess(i, f, im, conts, sum(giventargets[i], []), lang, vendor)


def group_given_contests(t, paths, giventargets, contests, flip, vendor, lang_map={}, NPROC=None):
    global tmp, flipped
    # print "ARGUMENTS", (t, paths, giventargets, lang_map)
    # print 'giventargets', giventargets
    # print lang_map
    if t[-1] != '/':
        t += '/'
    flipped = flip
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    # os.popen("rm -r "+tmp.replace(" ", "\\ ")+"*")
    if NPROC is None:
        NPROC = mp.cpu_count()
    if NPROC == 1:
        print "(Info) Using 1 process for group_given_contests"
        ballots = []
        args = [(vendor, lang_map, giventargets, x)
                for x in enumerate(zip(paths, contests))]
        for arg in args:
            ballots.append(group_given_contests_map(arg))
    else:
        print "(Info) Using {0} processors for group_given_contests".format(NPROC)
        pool = mp.Pool(NPROC)
        args = [(vendor, lang_map, giventargets, x)
                for x in enumerate(zip(paths, contests))]
        # print paths, giventargets, contests
        # print paths[11], giventargets[11], contests[11]
        # exit(0)
        ballots = pool.map(group_given_contests_map, args)
        pool.close()
        pool.join()
        # ballots = map(group_given_contests_map, args)
    print "WORKING ON", ballots
    return ballots, final_grouping(ballots, giventargets, paths, lang_map)


@util.pdb_on_crash
def final_grouping(ballots, giventargets, paths, langs, t=None):
    global tmp
    if t is not None:
        tmp = t
    lookup = dict((x, i) for i, x in enumerate(paths))
    if langs:
        languages = dict((idx, langs[imP]) for imP, idx in lookup.iteritems())
    else:
        languages = {}
    # languages = dict((lookup[k],v) for k,v in langs.items())
    print "RUNNING FINAL GROUPING"
    # pickle.dump((ballots, giventargets), open("/tmp/aaa", "w"))
    ballots = merge_contests(ballots, giventargets)
    print "NOW EQU CLASSES"
    # print ballots
    # pickle.dump((ballots, languages), open("/tmp/aaa", "w"))
    return equ_class(ballots, languages)


def sort_nicely(l):
    """ Sort the given list in the way that humans expect. Does an inplace sort.
    From:
        http://www.codinghorror.com/blog/2007/12/sorting-for-humans-natural-sort-order.html
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)

tmp = "tmp"
