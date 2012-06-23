import PIL
from PIL import Image, ImageDraw
import os
import time

def color(data, size):
    rows,cols = size
    ht = [0]*len(data)
    def dfs(i, fill):
        stack = [i]
        j = 0
        while stack != []:
            i = stack.pop()
            if ht[i] != 0 or data[i] < 250: continue
            j += 1
            ht[i] = fill
            if (i+1)%cols != 0:
                stack.append(i+1)
            if i%cols != 0:
                stack.append(i-1)
            if i-cols > 0:
                stack.append(i-cols)
            if i+cols < len(data):
                stack.append(i+cols)
        return j

    c = 1
    draws = []
    for i,each in list(enumerate(data))[::5]:
        if each >= 250 and ht[i] == 0:
            num = dfs(i, c)
            draws.append((i,num))
            c += 1

    draws = [x for x in draws if x[1] > 5]

    return ht,draws

def drawColored(ht, draws, size):
    rows,cols = size
    colored = Image.new("RGB", size)
    colored.putdata([(hash('awrha'+str(x)+'kfawge')%255, 
                      hash('agawrh'+str(x)+'qoeigngb')%255, 
                      hash('qpigb'+str(x)+'woekghal')%255) for x in ht])
    drcolor = ImageDraw.Draw(colored)
    for pos,what in draws:
        drcolor.text((pos%cols, pos/cols), str(what))
    return colored


def fill(name, original, dosave=True, dotargets=False):
    name = name[:-4]
    original.draft("L", original.size)
    data = original.getdata()
    cols,rows = original.size
    print rows, cols

    ht,draws = color(data, (rows,cols))
    
    colored = drawColored(ht, draws, (cols,rows))
    if dosave:
        colored.save("tmp/"+name+"-0.png")

    def boundingbox(pos):
        c = ht[pos]
        stack = [pos]
        seen = {}
        l = u = 1<<30
        r = d = -1<<30
        lht = len(ht)
        while stack != []:
            pos = stack.pop()
            if pos in seen: continue
            if l > pos%cols: l = pos%cols
            if r < pos%cols: r = pos%cols
            if u > pos/cols: u = pos/cols
            if d < pos/cols: d = pos/cols
            seen[pos] = True
            for each in [1,-1,cols,-cols]:
                if 0 <= pos+each < lht and ht[pos+each] == c:
                    stack.append(pos+each)
        return l,u,r,d

    imgbox = Image.new("L", (cols,rows))
    imgbox.putdata(data)
    drimgbox = ImageDraw.Draw(imgbox)
    for i,each in enumerate(draws):
        l,u,r,d = boundingbox(each[0])
        drimgbox.text((each[0]%cols, each[0]/cols), str(i))
        drimgbox.rectangle((l,u,r,d))
    if dosave:
        imgbox.save("tmp/"+name+"-1.png")

    targets = []
    contests = []
    for box in draws:
        l,u,r,d = boundingbox(box[0])
        if r-l < 15 and d-u < 15 and 20 < box[1] < 40:
            targets.append((l,u,r,d))
        if r-l > 30 and d-u > 30 and box[1] > 1000 and (0.0+box[1])/((r-l+1)*(d-u+1)) > .3:
            if (r-l+1)*(d-u+1) < len(data)/2:
                contests.append((l,u,r+1,d+1))

    def hastarget(x):
        ll, uu, rr, dd = x
        for (l,u,r,d) in targets:
            if ll < l < r < rr and uu < u < d < dd:
                return True
        return False

    filtercontests = [x for x in contests if hastarget(x)]

    contestbox = Image.new("L", (cols,rows))
    contestbox.putdata(data)
    drawcontest = ImageDraw.Draw(contestbox)

    for each in contests:
        drawcontest.rectangle(each, fill=200)
    for each in filtercontests:
        drawcontest.rectangle(each, fill=100)
    for each in targets:
        drawcontest.rectangle(each, fill=0)

    if dosave:
        for box in filtercontests:
            m = original.crop(box)
            n = name+"-"+str(box)[1:-1].replace(', ', 'x')
            m.save("tmp/contest-"+hex(hash("argioawrh"+n))[2:]+'-'+n+".png")
        

    if dosave:
        contestbox.save("tmp/"+name+"-2.png")

    if dotargets: return targets

    return filtercontests

    #os.popen("open tmp/"+name+"0.png tmp/"+name+"1.png")

def findtext(name, img, size, dosave=True):
    ht,draws = color(img, (size[1],size[0]))
    if dosave:
        drawColored(ht, draws, size).save("tmp/"+name+".colored.png")

    #print name
    lines = [0,0]+[255*size[0]-sum(img[x*size[0]:x*size[0]+size[0]]) for x in range(size[1])]+[0,0]
    
    im = Image.new("L", size)
    im.putdata(img)
    dr = ImageDraw.Draw(im)

    pos = []

    for i in range(2,len(lines)-3):
        if lines[i] < 100 or (lines[i-1]+lines[i+1]+lines[i-2]+lines[i+2])/lines[i] > 30:
            pos.append(i-2)
            dr.line((0, i-2, size[0], i-2))
    if dosave:
        im.save("tmp/lined-"+name)

    lines = lines[2:-2]
    adj = [0]+pos+[size[1]]
    startend = list(zip(adj, adj[1:]))

    extra = Image.new("RGB", size)
    extra.putdata([(x,x,x) for x in im.getdata()])
    dr = ImageDraw.Draw(extra)

    #print pos
    profile = []
    starts = []
    for start,end in startend:
        if end > start+4:
            starts.append((start,end))
            v = [[img[x*size[0]+y] for x in range(start+1,min(end+1,size[1]))] for y in range(size[0])]
            avg = [(0.0+sum(x))/(end-start) for x in v]
            #stdev = [sum([((0.0+y)-avg[i])**2 for y in x])**.5 for i,x in enumerate(v)]
            com = [(sum([i*(255-y) for i,y in enumerate(x)]))/(sum([255-y for y in x])+0.01) for x in v]
            #print start,end
            #print zip(com,[[i*(255-y) for i,y in enumerate(x)] for x in v],[sum([(255-y) for i,y in enumerate(x)]) for x in v])
            #com = [0 if x < 1.5 else 1 for x in com]
            #print "COM", com
            #print v,avg,stdev
            #print avg,stdev
            for i in range(size[0]):
                dr.point((i,start+1+int(round(com[i]))), fill=(255,0,0))#(255-int(avg[i])/2,int(avg[i])/2,0))
            profile.append((avg,[],com))
    if dosave:
        extra.save("tmp/col-lined-"+name)
    return starts,profile

def profeq(prof1, prof2):
    if len(prof1) != len(prof2): return False
    d2 = 0
    for line1,line2 in zip(prof1,prof2):
        fix = [a-b for a,b in zip(line1[2],line2[2]) if b != 0]
        fixv = sum(fix)/len(fix)
        cline2 = [x if x == 0 else x+fixv for x in line2[2]]
        t2 = sum([abs(x-y) for x,y in zip(line1[2], cline2)])
        d2 += t2
    return d2 < 20*len(prof1)


def makeDigraph(profiles):
    digraph = {}
    vlist = []

    for i in range(len(profiles)):
        digraph[i] = []


    for i,prof1 in enumerate(profiles):
    #for i,prof1 in enumerate([profiles[184]]):
        v = []
        v0 = []
        v2 = []
        for j,prof2 in enumerate(profiles[:i]):
        #for j,prof2 in enumerate([profiles[135]]):
        #for j,prof2 in enumerate([profiles[123]]):
            if len(prof2) != len(prof1) or j in digraph[i]:
                continue
            d2 = 0
            d0 = 0
            ct= 0
            #print len(prof1), len(prof2)
            for line1,line2 in zip(prof1,prof2):
                t0 = sum([abs(x-y) for x,y in zip(line1[0], line2[0])])
                fix = [a-b for a,b in zip(line1[2],line2[2]) if b != 0]
                fixv = sum(fix)/len(fix)
                #print fix
                #print 'avgerr =',fixv
                cline1 = [x if x == 0 else x-fixv for x in line2[2]]
                cline2 = [x if x == 0 else x+fixv for x in line2[2]]
                t2 = sum([abs(x-y) for x,y in zip(line1[2], cline2)])
                #t2 = sum([abs(x-y) for x,y in zip(line1[2], line2[2])])
    
                d2 += t2
                d0 += t0
                ct += 1
    
                #print map(int,[abs(x-y) for x,y in zip(line1[0], line2[0])])
                #print t0, t2
            v.append((d0/300+d2,j))
            v0.append((d0,j))
            v2.append((d2,j))
        vlist.append(v)
        #print i,names[i], sorted(v)
        #print i,"V0", sorted(v0)
        #print i,"V2", sorted(v2)
        #print "thresh", len(prof1)
        close = [x for score,x in sorted(v) if score < 20*len(prof1)]
        print i,names[i], close#, [x for s,x in sorted(v)]
    
        # Add them in, remove duplicates
        for each in close:
            if each not in digraph[i]:
                digraph[i] += [each]
            if i not in digraph[each]:
                digraph[each] += [i]
    return digraph

def components(graph):
    def dfsgraph(n):
        seen = {}
        stack = [n]
        while stack != []:
            vertex = stack.pop()
            if vertex in seen: continue
            seen[vertex] = True
            stack += graph[vertex]
        return seen.keys()
    sofar = {}
    comp = []
    for node in graph:
        if node not in sofar:
            lst = dfsgraph(node)
            comp.append(lst)
            for n in lst:
                sofar[n] = True
    return comp


if __name__ == "__main__":
        
    
    #each = "19_1.png"
    #if 1 == 1:
    for each in os.listdir("front/"):
        print each
        img = Image.open("front/"+each)
        fill(each, img)
    #exit(0)
    
    names = []
    imgs = []
    cpys = []
    for each in os.listdir("tmp/"):
        if 'contest' in each and not 'colored' in each and 'lined' not in each:
            names.append(each)
            im = Image.open("tmp/"+each)
            cpys.append(im.copy())
            imgs.append((list(im.getdata()),im.size))
    
    
    profiles = [None]*len(imgs)
    for i,(img1,size) in enumerate(imgs):
        #if not (i == 0 or i == 137): continue
        print "I IS", i, names[i]
        starts,prof = findtext(names[i], img1, size)
        #print starts
        profiles[i] = prof
    
    digraph = makeDigraph(profiles)
    
    
    # If I connects to J, and J connects to K, I doesn't need to check if it connects to K.
    
    #exit(0)
    
    for i,c in sorted(digraph.items()):
        print i, c
    
    
    """
    print "CROSS WEIGHT"
    for i,c in enumerate(components(digraph)):
        it = []
        for each1 in c:
            for each2 in c:
                if each1 in [z for w,z in vlist[each2]]:
                    it.append((each2,each1,vlist[each2][[z for w,z in vlist[each2]].index(each1)][0]))
                if each2 in [z for w,z in vlist[each1]]:
                    it.append((each1,each2,vlist[each1][[z for w,z in vlist[each1]].index(each2)][0]))
        print i,it
    """
    
    
    print "COMPONENTS"            
    
    for i,c in enumerate(components(digraph)):
        print "Group",i,"Components",[names[x] for x in c]
        
            
    
    for ind,bb in enumerate(components(digraph)):
        #print names[i], [names[j] for j in bb]
        res = Image.new("L", (sum([(cpys[i].size[0]) for i in bb]), max([(cpys[i].size[1]) for i in bb])))
        x = 0
        for i in bb:
            res.paste(cpys[i], (x, 0))
            x += cpys[i].size[0]
        res.save("tmp/group"+str(ind)+".png")
            
    
