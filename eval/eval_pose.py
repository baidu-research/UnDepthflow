from __future__ import division
import  sys
from glob import glob
from pose_evaluation_utils import *
from matplotlib import pyplot as plt


# Adopted from https://github.com/tinghuiz/SfMLearner
def eval_snippet(pred_dir, gt_dir):
    pred_files = glob(pred_dir + '/*.txt')
    ate_all = []
    for i in range(len(pred_files)):
        gtruth_file = gt_dir + "/" + os.path.basename(pred_files[i])
        if not os.path.exists(gtruth_file):
            continue
        ate = compute_ate(gtruth_file, pred_files[i])
        if ate == False:
            continue
        ate_all.append(ate)
    ate_all = np.array(ate_all)
    sys.stderr.write("ATE mean: %.4f, std: %.4f \n" % (np.mean(ate_all), np.std(ate_all)))

# Adopted from https://github.com/Huangying-Zhan/Depth-VO-Feat
class kittiEvalOdom():
  # ----------------------------------------------------------------------
  # poses: N,4,4
  # pose: 4,4
  # ----------------------------------------------------------------------
  def __init__(self, gt_dir):
    self.lengths= [100,200,300,400,500,600,700,800]
    self.num_lengths = len(self.lengths)
    self.gt_dir = gt_dir

  def loadPoses(self, file_name):
    # ----------------------------------------------------------------------
    # Each line in the file should follow one of the following structures
    # (1) idx pose(3x4 matrix in terms of 12 numbers)
    # (2) pose(3x4 matrix in terms of 12 numbers)
    # ----------------------------------------------------------------------
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    file_len = len(s)
    poses = {}
    for cnt, line in enumerate(s):
      P = np.eye(4)
      line_split = [float(i) for i in line.split(" ")]
      withIdx = int(len(line_split)==13)
      for row in xrange(3):
        for col in xrange(4):
          P[row, col] = line_split[row*4+col+ withIdx]
      if withIdx:
        frame_idx = line_split[0]
      else:
        frame_idx = cnt
      poses[frame_idx] = P
    return poses

  def trajectoryDistances(self, poses):
    # ----------------------------------------------------------------------
    # poses: dictionary: [frame_idx: pose]
    # ----------------------------------------------------------------------
    dist = [0]
    sort_frame_idx = sorted(poses.keys())
    for i in xrange(len(sort_frame_idx)-1):
      cur_frame_idx = sort_frame_idx[i]
      next_frame_idx = sort_frame_idx[i+1]
      P1 = poses[cur_frame_idx]
      P2 = poses[next_frame_idx]
      dx = P1[0,3] - P2[0,3]
      dy = P1[1,3] - P2[1,3]
      dz = P1[2,3] - P2[2,3]
      dist.append(dist[i]+np.sqrt(dx**2+dy**2+dz**2))  
    return dist

  def rotationError(self, pose_error):
    a = pose_error[0,0]
    b = pose_error[1,1]
    c = pose_error[2,2]
    d = 0.5*(a+b+c-1.0)
    return np.arccos(max(min(d,1.0),-1.0))

  def translationError(self, pose_error):
    dx = pose_error[0,3]
    dy = pose_error[1,3]
    dz = pose_error[2,3]
    return np.sqrt(dx**2+dy**2+dz**2)

  def lastFrameFromSegmentLength(self, dist, first_frame, len_):
    for i in xrange(first_frame, len(dist), 1):
      if dist[i] > (dist[first_frame] + len_):
        return i
    return -1

  def calcSequenceErrors(self, poses_gt, poses_result):
    err = []
    dist = self.trajectoryDistances(poses_gt)
    self.step_size = 10
    
    for first_frame in xrange(9, len(poses_gt), self.step_size):
      for i in xrange(self.num_lengths):
        len_ = self.lengths[i]
        last_frame = self.lastFrameFromSegmentLength(dist, first_frame, len_)

        # ----------------------------------------------------------------------
        # Continue if sequence not long enough
        # ----------------------------------------------------------------------
        if last_frame == -1 or not(last_frame in poses_result.keys()) or not(first_frame in poses_result.keys()):
          continue

        # ----------------------------------------------------------------------
        # compute rotational and translational errors
        # ----------------------------------------------------------------------
        pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
        pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
        pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

        r_err = self.rotationError(pose_error)
        t_err = self.translationError(pose_error)

        # ----------------------------------------------------------------------
        # compute speed 
        # ----------------------------------------------------------------------
        num_frames = last_frame - first_frame + 1.0
        speed = len_/(0.1*num_frames)

        err.append([first_frame, r_err/len_, t_err/len_, len_, speed])
    return err
    
  def saveSequenceErrors(self, err, file_name):
    fp = open(file_name,'w')
    for i in err:
      line_to_write = " ".join([str(j) for j in i])
      fp.writelines(line_to_write+"\n")
    fp.close()

  def computeOverallErr(self, seq_err):
    t_err = 0
    r_err = 0

    seq_len = len(seq_err)

    for item in seq_err:
      r_err += item[1]
      t_err += item[2]
    ave_t_err = t_err / seq_len
    ave_r_err = r_err / seq_len
    return ave_t_err, ave_r_err 

  def plotPath(self, seq, poses_gt, poses_result):
    plot_keys = ["Ground Truth", "Ours"]
    fontsize_ = 20
    plot_num =-1
      
    poses_dict = {}
    poses_dict["Ground Truth"] = poses_gt
    poses_dict["Ours"] = poses_result

    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')

    for key in plot_keys:
      pos_xz = []
      # for pose in poses_dict[key]:
      for frame_idx in sorted(poses_dict[key].keys()):
        pose = poses_dict[key][frame_idx]
        pos_xz.append([pose[0,3], pose[2,3]])
      pos_xz = np.asarray(pos_xz)
      plt.plot(pos_xz[:,0], pos_xz[:,1], label = key)  
      
    plt.legend(loc = "upper right", prop={'size': fontsize_})
    plt.xticks(fontsize = fontsize_) 
    plt.yticks(fontsize = fontsize_) 
    plt.xlabel('x (m)',fontsize = fontsize_)
    plt.ylabel('z (m)',fontsize = fontsize_)
    fig.set_size_inches(10, 10)
    png_title = "sequence_{:02}".format(seq)
    plt.savefig(self.plot_path_dir +  "/" + png_title + ".pdf",bbox_inches='tight', pad_inches=0)
    # plt.show()

  def plotError(self, avg_segment_errs):
    # ----------------------------------------------------------------------
    # avg_segment_errs: dict [100: err, 200: err...]
    # ----------------------------------------------------------------------
    plot_y = []
    plot_x = []
    for len_ in self.lengths:
      plot_x.append(len_)
      plot_y.append(avg_segment_errs[len_][0])
    fig = plt.figure()
    plt.plot(plot_x, plot_y)
    plt.show()

  def computeSegmentErr(self, seq_errs):
    # ----------------------------------------------------------------------
    # This function calculates average errors for different segment.
    # ----------------------------------------------------------------------

    segment_errs = {}
    avg_segment_errs = {}
    for len_ in self.lengths:
      segment_errs[len_] = []
    # ----------------------------------------------------------------------
    # Get errors
    # ----------------------------------------------------------------------
    for err in seq_errs:
      len_ = err[3]
      t_err = err[2]
      r_err = err[1]
      segment_errs[len_].append([t_err, r_err])
    # ----------------------------------------------------------------------
    # Compute average
    # ----------------------------------------------------------------------
    for len_ in self.lengths:
      if segment_errs[len_] != []:
        avg_t_err = np.mean(np.asarray(segment_errs[len_])[:,0])
        avg_r_err = np.mean(np.asarray(segment_errs[len_])[:,1])
        avg_segment_errs[len_] = [avg_t_err, avg_r_err]
      else:
        avg_segment_errs[len_] = []
    return avg_segment_errs

  def eval(self, result_dir):
    error_dir = result_dir + "/errors"
    self.plot_path_dir = result_dir + "/plot_path"
    plot_error_dir = result_dir + "/plot_error"

    if not os.path.exists(error_dir):
      os.makedirs(error_dir)
    if not os.path.exists(self.plot_path_dir):
      os.makedirs(self.plot_path_dir)
    if not os.path.exists(plot_error_dir):
      os.makedirs(plot_error_dir)

    total_err = []

    ave_t_errs = []
    ave_r_errs = []

    for seq in self.eval_seqs:
      self.cur_seq = seq
      file_name = seq + ".txt"

      poses_result = self.loadPoses(result_dir+"/"+file_name)
      poses_gt = self.loadPoses(self.gt_dir + "/" + file_name)
      self.result_file_name = result_dir+file_name

      # ----------------------------------------------------------------------
      # compute sequence errors
      # ----------------------------------------------------------------------
      seq_err = self.calcSequenceErrors(poses_gt, poses_result)
      self.saveSequenceErrors(seq_err, error_dir + "/" + file_name)

      # ----------------------------------------------------------------------
      # Compute segment errors
      # ----------------------------------------------------------------------
      avg_segment_errs = self.computeSegmentErr(seq_err)

      # ----------------------------------------------------------------------
      # compute overall error
      # ----------------------------------------------------------------------
      ave_t_err, ave_r_err = self.computeOverallErr(seq_err)
      sys.stderr.write("Sequence: " + seq + "\n")
      sys.stderr.write("Average translational RMSE (%%): %.2f \n" % (ave_t_err*100))
      sys.stderr.write("Average rotational error (deg/100m): %.2f \n" % (ave_r_err/np.pi * 180 *100))
      ave_t_errs.append(ave_t_err)
      ave_r_errs.append(ave_r_err)
