import math
import os
import subprocess
import uuid

from datetime import timedelta, datetime


class UtilsVideo:
    FFMPEG_PATH = 'ffmpeg'
    FFPROBE_PATH = 'ffprobe'

    @staticmethod
    def clip_video(video_path, start_at=None, end_at=None):
        working_path = os.path.dirname(video_path)
        output_file = working_path + '/' + str(uuid.uuid4()) + '.mp4'
        options = ''
        if start_at is not None:
            options = ' -ss ' + str(start_at)
        if end_at is not None:
            options += ' -to ' + str(end_at)

        command = UtilsVideo.FFMPEG_PATH + ' -i ' + video_path + options + ' -y -c copy ' + output_file
        p = subprocess.Popen(command, shell=True, close_fds=True, bufsize=-1)
        p.communicate()
        p.kill()
        p.terminate()
        p.wait()
        if p.stdout:
            p.stdout.close()

        if p.stdin:
            p.stdin.close()

        if p.stderr:
            p.stderr.close()

        if not os.path.exists(output_file):
            return None

        return output_file

    def grab_frame(rtsp_url, output_file):
        command = UtilsVideo.FFMPEG_PATH + ' -y -frames 1 ' + output_file + ' -rtsp_transport tcp -i ' + rtsp_url
        p = subprocess.Popen(command, shell=True, close_fds=True, bufsize=-1)
        p.communicate()
        p.kill()
        p.terminate()
        p.wait()
        if p.stdout:
            p.stdout.close()

        if p.stdin:
            p.stdin.close()

        if p.stderr:
            p.stderr.close()

    @staticmethod
    def concat_videos(video_files):
        working_path = os.path.dirname(video_files[0])
        merged_files = working_path + '/merged_files.txt'
        output_file = working_path + '/output.mp4'
        with open(merged_files, 'w') as f:
            for video_file in video_files:
                f.write("file '%s'\n" % video_file)

        command = UtilsVideo.FFMPEG_PATH + ' -f concat -safe 0 -i ' + merged_files + ' -c copy ' + output_file
        p = subprocess.Popen(command, shell=True)
        p.communicate()
        p.kill()
        p.terminate()
        p.wait()
        if p.stdout:
            p.stdout.close()

        if p.stdin:
            p.stdin.close()

        if p.stderr:
            p.stderr.close()

        if not os.path.exists(output_file):
            return None

        return output_file

    @staticmethod
    def get_video_info(file):
        p = subprocess.Popen([UtilsVideo.FFPROBE_PATH, '-show_format', '-pretty', '-loglevel', 'quiet', file],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, bufsize=-1)
        out, err = p.communicate()
        result = {'duration': -1, 'creation_time': None, 'modified_time': None}
        items = out.decode('ascii').splitlines()
        for item in items:
            if 'duration' in item:
                time_parts = item.split('=')[1].split(':')
                if len(time_parts) == 3:
                    result['duration'] = math.ceil(timedelta(hours=float(time_parts[0]), minutes=float(time_parts[1]),
                                                             seconds=float(time_parts[2])).total_seconds())

        result['modified_time'] = datetime.fromtimestamp(os.stat(file).st_mtime)
        if result['duration'] > 0:
            result['creation_time'] = result['modified_time'] - timedelta(seconds=result['duration'])

        p.kill()
        p.terminate()
        p.wait()
        if p.stdout:
            p.stdout.close()

        if p.stdin:
            p.stdin.close()

        if p.stderr:
            p.stderr.close()

        return result

    @staticmethod
    def convert_h264_2_mp4(video_file, output_path):
        command = UtilsVideo.FFMPEG_PATH + ' -i ' + video_file + ' -c copy ' + output_path
        p = subprocess.Popen(command, shell=True)
        p.kill()
        p.terminate()
        p.wait()
        if p.stdout:
            p.stdout.close()

        if p.stdin:
            p.stdin.close()

        if p.stderr:
            p.stderr.close()

        if not os.path.exists(output_path):
            return None

        return output_path
