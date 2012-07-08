from PIL import Image, ImageDraw
import os
from random import random
from collections import Counter

black = 200

do_save = True
export = True

def num2pil(img):
    pilimg = Image.new("L", (len(img[0]), len(img)))
    pilimg.putdata([item for sublist in img for item in sublist])
    return pilimg

def load_pil(path):
    pilimg = Image.open(path)
    pilimg = pilimg.convert("L")
    return pilimg

def load_num(path="", pilimg=None):
    if pilimg == None:
        pilimg = load_pil(path)
    width, height = pilimg.size
    data = list(pilimg.getdata())
    data = [data[x:x+width] for x in range(0,width*height,width)]
    #print width, height
    return data

def find_lines(data):
    height, width = len(data), len(data[0])
    def extend_ud(point):
        y, x = point
        while 0 < y and data[y][x] < black: y -= 1
        upper_y = y
        y, x = point
        low = len(data)
        while y < low and data[y][x] < black: y += 1
        lower_y = y
        return upper_y, lower_y
    def extend_lr(point):
        y, x = point
        while 0 < x and data[y][x] < black: x -= 1
        left_x = x
        y, x = point
        right = len(data[y])
        while x < right and data[y][x] < black: x += 1
        right_x = x
        return left_x, right_x

    def full_extend_ud(point):
        y, x = point
        l,r = extend_lr((y,x))
        x = (l+r)/2 if l+r<20 else x
        u1,d1 = extend_ud((y,x))
        return u1,d1

    def full_extend_lr(point):
        y, x = point
        u,d = extend_ud((y,x))
        y = (u+d)/2 if u+d<20 else y
        l1,r1 = extend_lr((y,x))
        return l1,r1

    LST = []
    def full_extend_lr_2(point):
        u,d = extend_ud(point)
        if d-u < 20: y = (u+d)/2
        else: y = point[0]
        point = (y,point[1])

        q = [point[0]]
        x = point[1]-1
        while q and x > 0:
            q = list(set([dy+y for y in q for dy in [-1, 0, 1] if 0 <= dy+y < height and data[dy+y][x] < black]))
            #LST.extend([(x,y) for y in q])
            x -= 1
        l = x
        q = [point[0]]
        x = point[1]+1
        while q and x < width:
            q = list(set([dy+y for y in q for dy in [-1, 0, 1] if 0 <= dy+y < height and data[dy+y][x] < black]))
            #LST.extend([(x,y) for y in q])
            x += 1
        r = x

        return l,r

    def full_extend_ud_2(point):
        l,r = extend_lr(point)
        if r-l < 20: x = (l+r)/2
        else: x = point[1]
        point = (point[0],x)

        q = [point[1]]
        y = point[0]-1
        while q and y > 0:
            q = list(set([dx+x for x in q for dx in [-1, 0, 1] if 0 <= dx+x < width and data[y][dx+x] < black]))
            #LST.extend([(x,y) for y in q])
            y -= 1
        u = y
        q = [point[1]]
        y = point[0]+1
        while q and y < height:
            q = list(set([dx+x for x in q for dx in [-1, 0, 1] if 0 <= dx+x < width and data[y][dx+x] < black]))
            #LST.extend([(x,y) for y in q])
            y += 1
        d = y

        return u,d

    foundy = {}
    foundx = {}
    lines = []
    YSKIP = 15
    XSKIP = 40
    for y in range(0,height,1):
        for x in range(0,width,1) if y%YSKIP == 0 else range(0,width,XSKIP):
            if not data[y][x] < black: 
                #data[y][x] = black
                continue
            if True:
                #print 'black at', y,x
                
                if y%YSKIP == 0 and (y/3,x/3) not in foundy:
                    u,d = full_extend_ud((y,x))
                    if d-u > 30:
                        u,d = full_extend_ud_2((y,x))
                        if d-u > 30:
                            for dx in range(-10, 10, 3):
                                for q in range(u,d):
                                    foundy[q/3,(x+dx)/3] = True
                            lines.append(("V", (x-3,u,x+3,d)))

                if x%XSKIP == 0 and (y/3,x/3) not in foundx:
                    #print 'h', newy, y, x
                    l,r = full_extend_lr_2((y,x))
                    if r-l > 30:
                        for dy in range(-10, 10, 3):
                            for q in range(l,r):
                                foundx[(y+dy)/3,q/3] = True
                        #print 'line starting from', x, y, data[y][x]
                        #LST.append((x-3,y-3,x+3,y+3))
                        #LST.append((l,y,r,y))
                        lines.append(("H", (l,y-3,r+10,y+3)))
            """
            except Exception as e:
                print e
                pass
            """
    
    if do_save:
        num2pil(data).save(tmp+"/it.png")
        ct = Counter(LST)
        im = Image.new("RGB", (width, height), (255,255,255))
        d = ImageDraw.Draw(im)
        LST = list(set(LST))
        for each in LST:
            if len(each) == 4:
                d.rectangle(each, fill=(0,0,0))
            else:
                d.point(each, fill=(ct[each],0,0))                
        im.save(tmp+"/asdf.png")

        im = Image.new("L", (width, height), 255)
        d = ImageDraw.Draw(im)
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] < black:
                    d.point((j,i), fill=0)
        im.save(tmp+"/asdf2.png")
    

    #print len(lines)
    #print lines
    return lines


def intersect(line1, line2):
    top = max(line1[1], line2[1])
    bottom = min(line1[3], line2[3])
    left = max(line1[0], line2[0])
    right = min(line1[2], line2[2])
    if bottom > top and right > left:
        return left, top, right, bottom
    else:
        return None
def intersect1(line1, line2):
    top = max(line1[1], line2[1])
    bottom = min(line1[3], line2[3])
    left = max(line1[0], line2[0])
    right = min(line1[2], line2[2])
    if bottom > top and right > left:
        return left, top, right, bottom
    else:
        return None
def union(line1, line2):
    top = min(line1[1], line2[1])
    bottom = max(line1[3], line2[3])
    left = min(line1[0], line2[0])
    right = max(line1[2], line2[2])
    return left, top, right, bottom

def dfs(graph, start):
    s = [start]
    seen = {}
    while s != []:
        top = s.pop()
        if top in seen: continue
        seen[top] = True
        s += graph[top]
    return seen.keys()

def to_graph(lines, width, height):
    print width, height
    table = [[None]*(width+20) for _ in range(height+20)]
    equal = []
    for full in lines:
        if full[0] != 'H': continue
        _,(l,u,r,d) = full
        for x in range(l,r):
            for y in range(u,d):
                if table[y][x] != None:
                    equal.append((table[y][x], full))
                else:
                    table[y][x] = full
    equal = list(set(equal))
    #print equal
    graph = {}
    for v1,v2 in equal:
        if v1 not in graph: graph[v1] = []
        if v2 not in graph: graph[v2] = []
        graph[v1].append(v2)
        graph[v2].append(v1)
    #print graph
    seen = {}
    new = []
    for el in graph.keys():
        if el in seen: continue
        makeequal = dfs(graph, el)
        for each in makeequal:
            seen[each] = True
        new.append((makeequal[0][0], reduce(union, [x[1] for x in makeequal])))
    for line in lines:
        if line not in seen:
            new.append(line)
    lines = new

        
    vertexes = dict((x, []) for _,x in lines)

    boxes = []
    for way1,line1 in lines:
        for way2,line2 in lines:
            if way1 != way2:
                if intersect(line1, line2):
                    boxes.append(intersect(line1, line2))
                    vertexes[line1].append(line2)
    return boxes,dict((k,v) for k,v in vertexes.items() if v != [])

def find_squares(graph):
    def dfs_square(stack, debug=False):
        if debug:
            print ".  "*len(stack), stack[-1]
        if len(stack) == 4:
            if stack[0] in graph[stack[-1]]:
                return [intersect(union(stack[0],stack[2]), union(stack[1],stack[3]))]
            return [None]
        res = []
        for vertex in graph[stack[-1]]:
            if vertex in stack: continue
            res += dfs_square(stack+[vertex], debug)
        return res

    return list(set([x for x in sum([dfs_square([start]) for start in graph], []) if x]))

def do_extract(name, img, squares, giventargets):
    def area(x): 
        if x == None: return 0
        return (x[2]-x[0])*(x[3]-x[1])

    targets = [x for x in giventargets]
    avg_targ_area = sum(map(area,targets))/len(targets)
    squares = [x for x in squares if area(x) > avg_targ_area*2]
    """
    ## Begin hack since I have to deal with my own targets
    bigest_box = max(squares, key=area)
    squares = [x for x in squares if intersect(x, bigest_box) == x]
    print len(squares), 'remain'
    ## End hack
    
    targets = [x for x in squares if area(x) < 10000 and 20 < x[3]-x[1] < 80 and 40 < x[2]-x[0] < 80]
    
    lgraph = [(x,y) for x in targets for y in targets if intersect(x,y) != None]
    #print lgraph
    graph = reduce(lambda p,c: p[c[0]].append(c[1]) or p, lgraph, dict((x,[]) for x in targets))
    #print graph
    #print 'before', targets
    targets = []
    while graph != {}:
        tounion = dfs(graph, graph.keys()[0])
        targets.append(reduce(union,tounion))
        for k in tounion:
            del graph[k]
    #print 'after', targets
    """
    contests = []

    #print "T", targets
    for sq in sorted(squares, key=area):
        if sq in targets: continue
        inside = [t for t in targets if area(intersect(sq, t)) == area(t)]
        if inside != []:
            # Check if there are other targets inside of this contest bounding box.
            otherinside = [t for t in giventargets if t not in inside and area(intersect(sq, t)) == area(t)]
            if otherinside != []:
                print "HAD AN ERROR ON", name
            contests.append(sq)
            targets = [x for x in targets if x not in inside]

    #print "C", contests
    for cont in contests:
        if export:
            im = img.crop(cont)
            name = tmp+"/"+str(sum(im.histogram()[:100]))+".png"
            im = img.crop(cont)
            im.save(name)

    if do_save:
        new = Image.new("RGB", img.size, (255, 255, 255))
        imd = ImageDraw.Draw(new)
        for box in contests:
            imd.rectangle(box, outline=(0,0,255))
        print "GIVEN", giventargets
        for box in giventargets:
            imd.rectangle(box, fill=(255,0,0))
        new.save(tmp+"/qqq.png")

    return contests

    #print targets, contests
    #os.popen("open tmp/*")
    #exit(0)
        

def extract_contest(image_path, giventargets):
    #Image.open(image_path).save(tmp+"/"+image_path.split("/")[-1][:-4]+"-orig.png")
    data = load_num(image_path)
    lines = find_lines(data)
    boxes, graph = to_graph(lines, len(data[0]), len(data))
    squares = find_squares(graph)
    squares = sorted(squares, key=lambda x: -(x[2]-x[0])*(x[3]-x[1]))
    #print lines
    #print squares

    if do_save:
        show = num2pil(data)
        new = load_pil(image_path).copy().convert("RGB")#Image.new("RGB", show.size, (255, 255, 255))
        imd = ImageDraw.Draw(new)
        for line in [x[1] for x in lines]:
            imd.rectangle(line, outline=(0,0,0))
        for line in boxes:
            imd.rectangle(line, fill=(0,0,255))
    
        print len(squares), "NUM"
    
        new.save(tmp+"/"+image_path.split("/")[-1][:-4]+"-line.png")
    
        new = Image.new("RGB", show.size, (255, 255, 255))
        imd = ImageDraw.Draw(new)
        for line in graph:
            imd.rectangle(line, outline=(0,0,0))
        for line in boxes:
            imd.rectangle(line, fill=(0,0,255))
    
        print len(squares), "NUM"
    
        new.save(tmp+"/"+image_path.split("/")[-1][:-4]+"-line-2.png")
    
        for l,u,r,d in squares:
            c = (int(random()*255), int(random()*255), int(random()*255))
            imd.rectangle((l,u,r,d), fill=c)
        new.save(tmp+"/"+image_path.split("/")[-1][:-4]+"-box.png")

    return data, do_extract(image_path.split("/")[-1], 
                            load_pil(image_path), squares, giventargets)

def ballot_preprocess(i, f, image, contests, targets, lang):
    print 'making', f, f.split("/")[-1].split(".")[0]
    sub = os.path.join(tmp+"", f.split("/")[-1].split(".")[0]+"-dir")
    os.mkdir(sub)
    res = []
    for c in contests:
        os.mkdir(os.path.join(sub, "-".join(map(str,c))))
        t = compare_preprocess(lang, os.path.join(sub, "-".join(map(str,c))), 
                               image, c, targets)
        res.append((i, c, t))
    print "RESULTING", res
    return res


def compare_preprocess(lang, path, image, contest, targets):
    #print contest, targets
    #print [intersect(contest, x) for x in targets]
    #print 'all', targets
    targets = [x for x in targets if intersect(contest, x) == x]
    l,u,r,d = contest
    cont_area = load_num(pilimg=num2pil(image).crop((l+10,u+10,r-10,d-10)))
    print "TEXT FOR", contest
    #print "bottom of box", d
    #print 'targets', targets
    tops = [0]+sorted([a[1]-u-10 for a in targets])+[d]
    #print contest
    #print "USING", tops
    blocks = []
    for upper,lower in zip(tops, tops[1:]):
        print "POS", upper, lower
        img = num2pil(cont_area[upper:lower])
        name = os.path.join(path, str(upper)+".tif")
        img.save(name)
        os.popen("tesseract %s %s -l %s"%(name, name, lang))
        istarget = (upper != 0)
        if os.path.exists(name+".txt"):
            print "THIS BLOCK GOT", open(name+".txt").read().decode('utf8')
            blocks.append((istarget, open(name+".txt").read().decode('utf8')))
        else:
            print "-"*40
            print "OCR FAILED"
            print "-"*40
            blocks.append((istarget, ""))
            
    
    print blocks
    return blocks

memo = {}
def row_dist(a, b):
    global memo
    if a == b: return 0
    if (a,b) in memo: 
        return memo[a,b]
    table = [[0]*(len(b)+2) for _ in range(len(a)+2)]
    for i in range(len(a)+2): table[i][0] = i
    for i in range(len(b)+2): table[0][i] = i
 
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            table[i][j] = min(table[i-1][j] + 1,
                              table[i][j-1] + 1,
                              table[i-1][j-1] + (a[i-1] != b[j-1]))
    memo[a,b] = memo[b,a] = table[-2][-2]
    return table[-2][-2]

count = 0
def compare(otexts1, otexts2):
    print 'running with1', otexts1
    print 'running with2', otexts2
    # text -> length
    texts1 = dict((y, len(y)) for x,y in otexts1)
    texts2 = dict((y, len(y)) for x,y in otexts2)
    # text -> target true/false
    istargs1 = dict((y,x) for x,y in otexts1)
    istargs2 = dict((y,x) for x,y in otexts2)
    # Text associated with targets only
    targtext1 = [y for x,y in otexts1 if x]
    targtext2 = [y for x,y in otexts2 if x]
    size = sum(texts1.values())+sum(texts2.values())
    weights = sorted([(row_dist(a,b),a,b) for a in texts1 for b in texts2])

    print "11111"
    for each in enumerate(targtext1): print each
    print "22222"
    for each in enumerate(targtext2): print each

    val = 0
    matching = []
    while texts1 != {} and texts2 != {}:
        for weight,a,b in weights:
            if a in texts1 and b in texts2 and istargs1[a] == istargs2[b]:
                if istargs1[a] == True:
                    print 'together', targtext1.index(a), targtext2.index(b)
                    matching.append((targtext1.index(a),
                                     targtext2.index(b)))
                val += weight
                del texts1[a]
                del texts2[b]
                break
    print "MATCHING", matching
    return float(val+sum(texts1.values())+sum(texts1.values()))/size, matching

def first_pass(contests):
    ht = {}
    for each in contests:
        if len(each[2]) not in ht: ht[len(each[2])] = []
        ht[len(each[2])].append(each)
    return ht.values()

def merge_equal(contests):
    sets = []
    print len(contests)
    print "CONTS", contests
    for i,each in enumerate(contests):
        print 'on', i, '#', len(sets)
        found = False
        for s in sets:
            # get a representitive, then get the non-matching part, then the text
            score, matching = compare(s[0][0][2], each[2])
            #score = compare(s[0][2], each[2])
            if score < .2:
                s.append((each, matching))
                #s.append(each)
                found = True
                break
        if not found:
            sets.append([(each, list(zip(range(len(each[2])), range(len(each[2])))))])
            #sets.append([each])
    return sets
    for s in sets:
        print '='*80
        print 'NEW EQU CLASS'
        print '='*80
        for ss in s:
            print "-"*40
            print ss[0]+"."+str(ss[1])
            
def equ_class(contests):
    #print map(len, contests)
    #print contests
    contests = sum(contests, [])
    #print contests
    groups = first_pass(contests)
    # Each group is known to be different.
    result = []
    for group in groups:
        result += merge_equal(group)
    print "RETURNING", result
    return result

def merge_contests(ballot_data, fulltargets):
    new_data = []
    for ballot, targets in zip(ballot_data, fulltargets):
        print 'next'
        new_ballot = []
        for group in targets:
            print 'targs is', group
            equal = [i for t in group for i,(_,bounding,_) in enumerate(ballot) if intersect(t, bounding)]
            equal_uniq = []
            for e in equal:
                if e not in equal_uniq: equal_uniq.append(e)
            print 'add', equal_uniq
            merged = sum([ballot[x][2] for x in equal_uniq],[])
            new_ballot.append((ballot[equal[0]][0], [ballot[x][1] for x in list(set(equal))], merged))
        new_data.append(new_ballot)
    print new_data
    return new_data
            

def do_grouping(t, paths, giventargets, lang_map = {}):
    global tmp
    print "ARGUMENTS", (t, paths, giventargets, lang_map)
    print 'giventargets', giventargets
    if t[-1] != '/': t += '/'
    tmp = t
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    os.popen("rm -r "+tmp+"*")
    ballots = []
    print "LEN", len(giventargets), len(paths)
    for i,f in enumerate(paths):
        print f
        im, contests = extract_contest(f, sum(giventargets[i],[]))
        lang = lang_map[f] if f in lang_map else 'eng'
        get = ballot_preprocess(i, f, im, contests, sum(giventargets[i],[]), lang)
        ballots.append(get)
    print "WORKING ON", ballots
    ballots = merge_contests(ballots, giventargets)
    return equ_class(ballots)
