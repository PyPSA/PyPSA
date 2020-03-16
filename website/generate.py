# make the code as Python 3 compatible as possible
from __future__ import print_function, division, absolute_import

import os, sys


hierarchy = [["index.html","home"],
             ["download/index.html","download"],
             ["examples/index.html","examples"],
             ["doc/index.html","documentation"],
             ["publications/index.html","publications"],
             ["animations/index.html","animations"],
             ["forum/index.html","forum"],
]



def process_org(org_name):
    """Build HTML page and write it."""

    html_name = org_name[:-4] + ".html"

    root_path = org_name.count("/")*"../"

    hierarchy_path = []

    def find_path(layer,path,hierarchy_path):
        if len(path) > 0 and path[-1] == html_name:
            hierarchy_path += path
        else:
            for item in layer:
                new_path = path + [item[0]]
                if len(item) > 2:
                    new_layer = item[2]
                else:
                    new_layer = []
                find_path(new_layer,new_path,hierarchy_path)

    find_path(hierarchy,[],hierarchy_path)

    hierarchy_path.append("")

    layer = hierarchy

    name = ""

    menu = ""

    for n,item in enumerate(hierarchy_path):

        bare_layer = [i[0] for i in layer]

        filled_layer = [[i[0],i[0][:i[0].rfind(".")]] if len(i) < 2 else i for i in layer]

        menu += "<ul id=\"{}menu\">\n".format("sub"*(n+1))

        menu += "".join(["<li><a href=\"{}\">{}</a></li>\n".format(root_path + i[0],i[1]) if i[0] != item and i[0] != html_name else "<li><span class=\"current\">{}</span></li>\n".format(i[1]) for i in filled_layer])

        menu += "</ul>\n"

        if item != "":
            full_item = layer[ bare_layer.index(item)]
            for i in filled_layer:
                if i[0] == html_name:
                    name = i[1]
        else:
            full_item = []

        if len(full_item) < 3:
            break

        layer = full_item[2]


    html = """<!DOCTYPE html
PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"
\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">
<html>
<head>
<title>pypsa | {}</title>
<link rel=\"stylesheet\" type=\"text/css\" href=\"{}\" />
</head>
<body>
<div id="outer_box">
<div id="header">

{}

""".format(name, root_path + "theme.css", menu)


    html += "\n\n</div>\n\n<div id=\"main\">\n\n"


    command = "emacs {} --batch -f org-html-export-to-html --kill".format(org_name)

    os.system(command)

    f = open(html_name,"r")

    org_html = f.read()

    f.close()


    if "outer_box" in org_html:
        print("File is already processed, skipping")
        return

    start_string = '<div id="content">'

    if start_string not in org_html:
        print("Start string not found, skipping")
        return


    end_string = '<div id="postamble" class="status">'
    if end_string not in org_html:
        print("End string not found, skipping")
        return

    new = org_html[org_html.find(start_string)+len(start_string):org_html.find(end_string)]


    html  = html + new + "\n\n</body>\n</html>\n"

    f = open(html_name,"w")

    f.write(html)

    f.close()

    return html


for path, sub_dirs, file_names in os.walk("."):

    if "/old" in path:
        continue

    for file_name in file_names:
        if file_name[-4:] == ".org":
            full_name = os.path.join(path,file_name)[2:]
            print(full_name)
            process_org(full_name)
