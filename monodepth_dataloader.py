# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com
"""
Adopted from https://github.com/mrharicot/monodepth
Monodepth data loader.
Copyright © Niantic, Inc. 2018. Patent Pending.
All rights reserved.

================================================================================

This Software is licensed under the terms of the UCLB ACP-A Licence which allows
for non-commercial use only. For any other use of the software not covered by
the terms of this licence, please contact info@uclb.com

================================================================================

UCLB ACP-A Licence

This Agreement is made by and between the Licensor and the Licensee as defined
and identified below. 

1.  Definitions.
    In this Agreement (“the Agreement”) the following words shall have the
    following meanings:

    "Authors" shall mean C. Godard, O. Mac Aodha, G. Brostow
    "Licensee" Shall mean the person or organisation agreeing to use the
    Software in accordance with these terms and conditions.
    "Licensor" shall mean UCL Business PLC whose registered office is at The
    Network Building, 97 Tottenham Court Road, London W1T 4TP. UCL Business is a
    the technology transfer arm of University College London (UCL).
    "Owner" shall mean Niantic Inc., a company organised and existing under the
    laws of Delaware, whose principal place of business is at 2 Bryant Street,
    #220, San Francisco, 94105. Owner is a third party beneficiary of this
    Agreement and may enforce its terms as if it were a party to this Agreement.
    "Software" shall mean the MonoDepth Software in source code or object code
    form and any accompanying documentation. 

2.  License.
    2.1 The Licensor has all necessary rights to grant a licence under: (i)
    copyright and rights in the nature of copyright subsisting in the Software;
    and (ii) patent rights resulting from a patent application filed by the
    Licensor in the United Kingdom in connection with the Software. The Licensor
    grants the Licensee for the duration of this Agreement, a free of charge,
    non-sublicenseable, non-exclusive, non-transferable copyright and patent
    licence (in consequence of said patent application) to use the Software for
    non-commercial purpose only, including teaching and research at educational
    institutions and research at not-for-profit research institutions in
    accordance with the provisions of this Agreement. Non-commercial use
    expressly excludes any profit-making or commercial activities, including
    without limitation sale, licence, manufacture or development of commercial
    products, use in commercially-sponsored research, provision of consulting
    service, use for or on behalf of any commercial entity, and use in research
    where a commercial party obtains rights to research results or any other
    benefit. Any use of the Software for any purpose other than non-commercial
    research shall automatically terminate this Licence. 
    
    2.2 The Licensee is permitted to make modifications to the Software provided
    that distribution of such modifications is in accordance with Clause 3.

    2.3 Except as expressly permitted by this Agreement and save to the extent
    and in the circumstances expressly required to be permitted by law, the
    Licensee is not permitted to rent, lease, sell, offer to sell or loan the
    Software or its associated documentation.

3.  Redistribution and modifications
    3.1 The Licensee may reproduce and distribute copies of the Software only to
    this same GitHub repository with or without modifications, in source format
    only and provided that any and every distribution is accompanied by an
    unmodified copy of this Licence and that the following copyright notice is
    always displayed in an obvious manner: Copyright © Niantic, Inc. 2018. All
    rights reserved. 
    
    3.2 In the case where the Software has been modified, any distribution must
    include prominent notices indicating which files have been changed.
    
    3.3 The Licensee shall cause any work that it distributes or publishes, that
    in whole or in part contains or is derived from the Software or any part
    thereof (“Work based on the Software”), to be licensed as a whole at no
    charge to all third parties under the terms of this Licence.

4.  Duration.
    This Agreement is effective until the Licensee terminates it by destroying
    the Software and its documentation together with all copies. It will also
    terminate automatically if the Licensee fails to abide by its terms. Upon
    automatic termination the Licensee agrees to destroy all copies of the
    Software and its documentation.

5.  Disclaimer of Warranties.
    The Software is provided as is. To the maximum extent permitted by law,
    Licensor provides no warranties or conditions of any kind, either express or
    implied, including without limitation, any warranties or condition of title,
    non-infringement or fitness for a particular purpose. 

6.  Limitation of Liability.
    In no event shall the Licensor and/or Authors be liable for any direct,
    indirect, incidental, special, exemplary or consequential damages (including
    but not limited to, procurement of substitute goods or services; loss of
    use, data or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use of
    this Software, even if advised of the possibility of such damage.

7.  Indemnity.
    The Licensee shall indemnify the Licensor and/or Authors against all third
    party claims that may be asserted against or suffered by the Licensor and/or
    Authors and which relate to use of the Software by the Licensee or the
    Recipient.

8.  Intellectual Property.
    8.1 As between the Licensee and Licensor,copyright and all other
    intellectual property rights subsisting in or in connection with the
    Software and supporting information shall remain at all times the property
    of the Licensor but Licensee acknowledges and agrees that Owner is the owner
    of all right, title and interest in and to the Software. The Licensee shall
    acquire no rights in any such material except as expressly provided in this
    Agreement.

    8.2 No permission is granted to use the trademarks or product names of the
    Licensor or Owner except as required for reasonable and customary use in
    describing the origin of the Software and for the purposes of abiding by the
    terms of Clause 3.1.

    8.3 The Licensee shall promptly notify the Licensor, in sufficient detail,
    all improvements and new uses of the Software (“Improvements”). The Licensor
    and its affiliates shall have a non-exclusive, fully paid-up, royalty-free,
    irrevocable and perpetual licence under the Improvements for non-commercial
    academic research and teaching purposes.  

    8.4 The Licensee grants an exclusive first option to the Owner to be
    exercised by the Owner within three (3) years of the date of notification of
    the Improvements under Clause 8.3 to use any Improvements for commercial
    purposes on terms to be negotiated and agreed by Licensee and Owner in good
    faith within a period of six (6) months from the date of exercise of the
    said option (including without limitation any royalty share in net income
    from such commercialization payable to the Licensee, as the case may be).

9.  Acknowledgements.
    The Licensee shall acknowledge the Authors and use of the Software in the
    publication of any work that uses, or results that are achieved through, the
    use of the Software. The following citation shall be included in the
    acknowledgement: “Unsupervised Monocular Depth Estimation with Left-Right
    Consistency, by C. Godard, O Mac Aodha, G. Brostow, CVPR 2017.”

10. Governing Law.
    This Agreement shall be governed by, construed and interpreted in accordance
    with English law and the parties submit to the exclusive jurisdiction of the
    English courts.

11. Termination.
    Upon termination of this Agreement, the licenses granted hereunder will
    terminate and Sections 5, 6, 7, 8, 9, 10 and 11 shall survive any
    termination of this Agreement. 
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


def rescale_intrinsics(raw_cam_mat, opt, orig_height, orig_width):
    fx = raw_cam_mat[0, 0]
    fy = raw_cam_mat[1, 1]
    cx = raw_cam_mat[0, 2]
    cy = raw_cam_mat[1, 2]
    r1 = tf.stack(
        [fx * opt.img_width / orig_width, 0, cx * opt.img_width / orig_width])
    r2 = tf.stack([
        0, fy * opt.img_height / orig_height, cy * opt.img_height / orig_height
    ])
    r3 = tf.constant([0., 0., 1.])
    return tf.stack([r1, r2, r3])


def get_multi_scale_intrinsics(raw_cam_mat, num_scales):
    proj_cam2pix = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = raw_cam_mat[0, 0] / (2**s)
        fy = raw_cam_mat[1, 1] / (2**s)
        cx = raw_cam_mat[0, 2] / (2**s)
        cy = raw_cam_mat[1, 2] / (2**s)
        r1 = tf.stack([fx, 0, cx])
        r2 = tf.stack([0, fy, cy])
        r3 = tf.constant([0., 0., 1.])
        proj_cam2pix.append(tf.stack([r1, r2, r3]))
    proj_cam2pix = tf.stack(proj_cam2pix)
    proj_pix2cam = tf.matrix_inverse(proj_cam2pix)
    proj_cam2pix.set_shape([num_scales, 3, 3])
    proj_pix2cam.set_shape([num_scales, 3, 3])
    return proj_cam2pix, proj_pix2cam


def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0., 0., 1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics


def data_augmentation(im, intrinsics, out_h, out_w):
    # Random scaling
    def random_scaling(im, intrinsics):
        batch_size, in_h, in_w, _ = im.get_shape().as_list()
        scaling = tf.random_uniform([2], 1, 1.15)
        x_scaling = scaling[0]
        y_scaling = scaling[1]
        out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
        im = tf.image.resize_area(im, [out_h, out_w])
        fx = intrinsics[:, 0, 0] * x_scaling
        fy = intrinsics[:, 1, 1] * y_scaling
        cx = intrinsics[:, 0, 2] * x_scaling
        cy = intrinsics[:, 1, 2] * y_scaling
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    # Random cropping
    def random_cropping(im, intrinsics, out_h, out_w):
        # batch_size, in_h, in_w, _ = im.get_shape().as_list()
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
        offset_y = tf.random_uniform(
            [1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
        offset_x = tf.random_uniform(
            [1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
        im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h,
                                           out_w)
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2] - tf.cast(offset_x, dtype=tf.float32)
        cy = intrinsics[:, 1, 2] - tf.cast(offset_y, dtype=tf.float32)
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    im, intrinsics = random_scaling(im, intrinsics)
    im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
    return im, intrinsics


class MonodepthDataloader(object):
    """monodepth dataloader"""

    def __init__(self, opt):
        self.data_path = opt.data_dir
        self.opt = opt
        filenames_file = opt.train_file

        input_queue = tf.train.string_input_producer(
            [filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values

        # we load only one image for test, except if we trained a stereo model
        left_image_path = tf.string_join([self.data_path, split_line[0]])
        right_image_path = tf.string_join([self.data_path, split_line[1]])
        next_left_image_path = tf.string_join([self.data_path, split_line[2]])
        next_right_image_path = tf.string_join([self.data_path, split_line[3]])
        cam_intrinsic_path = tf.string_join([self.data_path, split_line[4]])

        left_image_o, orig_height, orig_width = self.read_image(
            left_image_path, get_shape=True)
        right_image_o = self.read_image(right_image_path)
        next_left_image_o = self.read_image(next_left_image_path)
        next_right_image_o = self.read_image(next_right_image_path)

        # randomly flip images
        do_flip = tf.random_uniform([], 0, 1)
        left_image = tf.cond(do_flip > 0.5,
                             lambda: tf.image.flip_left_right(right_image_o),
                             lambda: left_image_o)
        right_image = tf.cond(do_flip > 0.5,
                              lambda: tf.image.flip_left_right(left_image_o),
                              lambda: right_image_o)

        next_left_image = tf.cond(
            do_flip > 0.5,
            lambda: tf.image.flip_left_right(next_right_image_o),
            lambda: next_left_image_o)
        next_right_image = tf.cond(
            do_flip > 0.5, lambda: tf.image.flip_left_right(next_left_image_o),
            lambda: next_right_image_o)

        do_flip_fb = tf.random_uniform([], 0, 1)
        left_image, right_image, next_left_image, next_right_image = tf.cond(
            do_flip_fb > 0.5,
            lambda: (next_left_image, next_right_image, left_image, right_image),
            lambda: (left_image, right_image, next_left_image, next_right_image)
        )

        # randomly augment images
        #         do_augment  = tf.random_uniform([], 0, 0)
        #         image_list = [left_image, right_image, next_left_image, next_right_image]
        #         left_image, right_image, next_left_image, next_right_image = tf.cond(do_augment > 0.5, 
        #                                                                              lambda: self.augment_image_list(image_list), 
        #                                                                              lambda: image_list)

        left_image.set_shape([None, None, 3])
        right_image.set_shape([None, None, 3])
        next_left_image.set_shape([None, None, 3])
        next_right_image.set_shape([None, None, 3])

        raw_cam_contents = tf.read_file(cam_intrinsic_path)
        last_line = tf.string_split(
            [raw_cam_contents], delimiter="\n").values[-1]
        raw_cam_vec = tf.string_to_number(
            tf.string_split([last_line]).values[1:])
        raw_cam_mat = tf.reshape(raw_cam_vec, [3, 4])
        raw_cam_mat = raw_cam_mat[0:3, 0:3]
        raw_cam_mat = rescale_intrinsics(raw_cam_mat, opt, orig_height,
                                         orig_width)

        # Scale and crop augmentation
        #         im_batch = tf.concat([tf.expand_dims(left_image, 0), 
        #                          tf.expand_dims(right_image, 0),
        #                          tf.expand_dims(next_left_image, 0),
        #                          tf.expand_dims(next_right_image, 0)], axis=3)
        #         raw_cam_mat_batch = tf.expand_dims(raw_cam_mat, axis=0)
        #         im_batch, raw_cam_mat_batch = data_augmentation(im_batch, raw_cam_mat_batch, self.opt.img_height, self.opt.img_width)
        #         left_image, right_image, next_left_image, next_right_image = tf.split(im_batch[0,:,:,:], num_or_size_splits=4, axis=2)
        #         raw_cam_mat = raw_cam_mat_batch[0,:,:]

        proj_cam2pix, proj_pix2cam = get_multi_scale_intrinsics(raw_cam_mat,
                                                                opt.num_scales)

        # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 2048
        capacity = min_after_dequeue + 4 * opt.batch_size
        self.data_batch = tf.train.shuffle_batch([
            left_image, right_image, next_left_image, next_right_image,
            proj_cam2pix, proj_pix2cam
        ], opt.batch_size, capacity, min_after_dequeue, 10)

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image**random_gamma
        right_image_aug = right_image**random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack(
            [white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def augment_image_list(self, image_list):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        image_list = [img**random_gamma for img in image_list]

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        image_list = [img * random_brightness for img in image_list]

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones(
            [tf.shape(image_list[0])[0], tf.shape(image_list[0])[1]])
        color_image = tf.stack(
            [white * random_colors[i] for i in range(3)], axis=2)
        image_list = [img * color_image for img in image_list]

        # saturate
        image_list = [tf.clip_by_value(img, 0, 1) for img in image_list]

        return image_list

    def read_image(self, image_path, get_shape=False):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image = tf.cond(
            file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
            lambda: tf.image.decode_png(tf.read_file(image_path)))
        orig_height = tf.cast(tf.shape(image)[0], "float32")
        orig_width = tf.cast(tf.shape(image)[1], "float32")

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(
            image, [self.opt.img_height, self.opt.img_width],
            tf.image.ResizeMethod.AREA)

        if get_shape:
            return image, orig_height, orig_width
        else:
            return image
