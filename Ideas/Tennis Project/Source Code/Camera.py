
import cv2
import numpy as np

class Camera:

  CameraMatrix = [];
  DistCoeffs = [];
  Position = [];
  RotationVec = [];
  TranslationVec = [];
  CourtCorners = [];
  Homog = [];

  # HALF_COURT_X = 4.115;
  HALF_COURT_X = 5.485
  HALF_COURT_Z = 11.885;
  WORLD_POINTS = np.asarray([[-HALF_COURT_X, 0, -HALF_COURT_Z],
                             [ HALF_COURT_X, 0, -HALF_COURT_Z],
                             [ HALF_COURT_X, 0,  HALF_COURT_Z],
                             [-HALF_COURT_X, 0,  HALF_COURT_Z]], "float");

  def __init__(self, cameraName, courtCorners):
    if cameraName == "kyle":
      fx=1994.25368447834;
      fy=1988.65266798629;
      cx=968.573023612607;
      cy=511.585679422200;
      k1=0.0771110325943740;
      k2=-0.0596894545787290;
      p1=0.00178967197419077;
      p2=0.00123017525081653;
    elif cameraName == "megan":
      fx=1981.39204255929;
      fy=1973.70141739089;
      cx=980.523462971786;
      cy=551.217098728122;
      k1=0.0747612507420630;
      k2=-0.0683271738685350;
      p1=0.00240502474003212;
      p2=0.00199735586169493;
    else:
      raise ValueError("cameraName must be 'kyle' or 'megan'!")
      return;

    self.CourtCorners = courtCorners.copy();
    self.CameraMatrix = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]);
    self.DistCoeffs = np.asarray([ k1, k2, p1, p2 ]) #np.zeros((4,1)); # TODO: fill

    # FIND CAMERA POSITION
    imgCoords = np.transpose(courtCorners);
    _, rVec, tVec = cv2.solvePnP(self.WORLD_POINTS.reshape((4,1,3)), np.asarray(courtCorners.reshape((4,1,2)), dtype="float"), self.CameraMatrix, self.DistCoeffs,flags=cv2.SOLVEPNP_ITERATIVE);
    self.RotationVec = rVec.copy();
    self.Rotation = cv2.Rodrigues(rVec)[0];
    self.TranslationVec = tVec.copy();
    R_inv = np.transpose(self.Rotation);
    self.Position = - (np.matmul(R_inv,tVec))[:,0]
    #print self.Position



    # FIND MAPPING FROM CAM TO WORLD @ Y==0
    camPoints = np.zeros((4,2), dtype="float32");
    for i in range(0,4):
      pt = self.GetPinholePoint(self.CourtCorners[i,:]);
      camPoints[i,0] = pt[0]; # U coord
      camPoints[i,1] = pt[1]; # V coord
    worldPoints = self.WORLD_POINTS[:, [0,2]]
    self.Homog = cv2.findHomography(camPoints, worldPoints)[0];
    self.InvHomog = np.linalg.inv(self.Homog);

  # Undistort the pixel position and convert it to pinhole coordinates w/ focal length 1
  def GetPinholePoint(self, pt):
    pts = np.zeros((1,1,2));
    pts[0,0,0] = pt[0];
    pts[0,0,1] = pt[1];
    result = cv2.undistortPoints(pts, self.CameraMatrix, self.DistCoeffs);
    xy = np.asarray([result[0,0,0], result[0,0,1]]);
    return xy

  # Convert a point from pixel position to court position
  def ConvertPixelToCourtPosition(self, pt):
    pinholePt = self.GetPinholePoint(pt);
    # Convert a point from pinhole to court position
    pt2 = np.asarray([pinholePt[0], pinholePt[1], 1.0]);
    res = np.matmul(self.Homog, pt2);
    res /= res[2];
    return np.asarray([res[0], 0.0, res[1]]);

  # Convert 3d point to 2d pixel position
  def ConvertWorldToImagePosition(self, pt):
    # solve for court point
    pt1 = self.Position;
    pt2 = pt;
    t = - pt2[1] / (pt1[1] - pt2[1]);
    isectPt = pt1 * t + pt2 * (1-t);
    isectPt = np.asarray([isectPt[0], isectPt[2], 1.0]);
    isectPtPinhole = np.matmul(self.InvHomog, isectPt.reshape(3,1));
    isectPtPinhole /= isectPtPinhole[2];
    pxPt = cv2.projectPoints(isectPtPinhole.reshape(1,1,3), np.identity(3), np.asarray([0,0,0], dtype="float"), self.CameraMatrix, self.DistCoeffs)[0][0][0];
    pxPt = np.maximum(np.asarray([0,0]), pxPt);
    return np.asarray(pxPt, dtype="uint32")

  def GetRay(self, pxPosition):
    ctPos = self.ConvertPixelToCourtPosition(pxPosition)
    ctMinusCam = ctPos - self.Position;
    return (self.Position, ctMinusCam / np.linalg.norm(ctMinusCam));

# output:
# pt is the closest point between rays
# dist is the distance of the two rays at their nearest crossing
# D is the corresponding point on ray1
# E is the corresponding point on ray2
def IntersectRays(ray1, ray2):
  A = ray1[0];
  a = ray1[1];
  B = ray2[0];
  b = ray2[1];
  c = B - A;
  aa = np.dot(a,a);
  ac = np.dot(a,c);
  bb = np.dot(b,b);
  ab = np.dot(a,b);
  bc = np.dot(b,c);
  D = A + a * ((ac*bb - ab*bc) / (aa*bb - ab*ab));
  E = B + b * ((ab*ac - bc*aa) / (aa*bb - ab*ab));
  pt = (D+E)/2;
  dist = np.linalg.norm(D-E);
  return (pt, dist, D, E);


## TEST BENCH:
#from FindCourtCorners import CourtFinder
#cap = cv2.VideoCapture('../UntrackedFiles/stereoClip5_Megan.mov')
#_, frame = cap.read()
#cf = CourtFinder();
#cf.FindCourtCorners(frame);
#corners = np.asarray([[114,454],
#                      [766,444],
#                      [1805,835],
#                      [317,1034]]);
#kyleCam = Camera("kyle", corners);
#for i in range (0, 1):
#  print kyleCam.ConvertWorldToImagePosition(np.asarray([0,3,0]));
