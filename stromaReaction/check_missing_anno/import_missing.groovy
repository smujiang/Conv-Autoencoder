import qupath.lib.gui.scripting.QPEx
import qupath.lib.scripting.QP
import qupath.lib.roi.ROIs
import qupath.lib.geom.Point2
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.RectangleROI
import qupath.lib.objects.PathAnnotationObject
import static com.xlson.groovycsv.CsvParser.parseCsv

def imageData = QPEx.getCurrentImageData()
def server = QP.getCurrentImageData().getServer()
case_id = server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.'))
print(sprintf("Processing case %s", case_id))

txt_fn = "H:\\OvaryCancer\\StromaReaction\\Annotation" + File.separator +"check_mask_same_shape.txt"
fh = new File(txt_fn)
def txt_content = fh.getText('utf-8')
def data_iterator = parseCsv(txt_content, separator: ',', readFirstLine: false)
for (line in data_iterator) {
    def c_x = line[0] as Integer
    def c_y = line[1] as Integer
    line_case_ID = line[2]
    if (line_case_ID == case_id){
        def roi = new RectangleROI(c_x-200, c_y-200, 100, 100)
        def indicator = new PathAnnotationObject(roi)
        imageData.getHierarchy().addPathObject(indicator, false)
        QPEx.fireHierarchyUpdate()
    }
}

print("Done")





