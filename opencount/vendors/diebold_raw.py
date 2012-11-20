import cv
def decode(imagepath, template_path = "diebold-mark.jpg"):
    """
    input: path to image, path to template
    output: decoded string, coordinates of black markings in form(x_topleft, y_topleft, x_bottomright, y_bottomright)
    """
    image = cv.LoadImage(imagepath)
    template = cv.LoadImage(template_path)
    w,h = cv.GetSize(template)
    W,H = cv.GetSize(image)
    width = W - w + 1
    height = H - h + 1
    result = cv.CreateImage((width, height), 32, 1)
    cv.MatchTemplate(image, template, result, cv.CV_TM_CCOEFF_NORMED)
    coords = []
    for x in range(width):
        for y in range(height):
            if result[y,x] > .7:
                coords.append((x,y))
    min_distance = 2*w
    delete = []
    for index in range (1, len(coords)):
        distance = coords[index][0] - coords[index-1][0]
        if distance < w:
            delete.append(coords[index])
        if distance > w and distance < min_distance:
           min_distance = distance
    for d in delete:
        coords.remove(d)
    if len(coords) == 0:
        return None, None
    first = coords[0][0]
    coords = [(x - first,y) for x,y in coords]

    last = coords[-1][0]
    if last == 0.0:
        return None, None
    spacing = last / 33.0
    # 33 spots between first and last
    string = ["0" for k in range(0,34)]

    for x,y in coords:
        string[int((x+spacing*.2) // spacing)] = "1"


    coords = [(x,y,x+w,y+h) for x,y in coords]
    print "".join(string)
    print coords
    return string, coords
    








