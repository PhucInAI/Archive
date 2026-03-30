class ServerTypes:
    API_SERVER = 'api-server'
    STREAMING_SERVER = 'streaming-server'
    FACE_SERVER = 'face-server'
    HUMAN_SERVER = 'human-server'
    LICENSE_PLATE_SERVER = 'license-plate-server'

    values = [API_SERVER, STREAMING_SERVER, FACE_SERVER, HUMAN_SERVER, LICENSE_PLATE_SERVER]

    @staticmethod
    def get_name(server_type):
        return server_type.replace('-', ' ').title()
