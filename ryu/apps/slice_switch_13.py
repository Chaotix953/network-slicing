from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, ether_types
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.lib import hub
from webob import Response
import json
import time


SLICE_CONFIG = {
    'URLLC': {
        'queue_id': 1,
        'priority': 100,
        'dscp': [46],
        'ports': [5001, 5002],
        'name': 'URLLC'
    },
    'mMTC': {
        'queue_id': 2,
        'priority': 10,
        'dscp': [8, 10],
        'ports': [6001, 6002],
        'name': 'mMTC'
    }
}

slice_api_instance_name = 'slicing_api_app'
url_base = '/slicing'


class SlicingSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(SlicingSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}

        self.slice_stats = {
            'URLLC': {
                'packet_count': 0,
                'byte_count': 0,
                'flow_count': 0,
                'last_update': time.time(),

                'prev_packet_count': 0,
                'prev_byte_count': 0,

                'throughput_mbps': 0.0,
                'avg_latency_ms': 0.0,
                'packet_loss_rate': 0.0,

                'tx_packets': 0,
                'tx_bytes': 0,

                # snapshots for delta computation
                'last_tx_packets_snapshot': None,
                'last_tx_bytes_snapshot': None
            },
            'mMTC': {
                'packet_count': 0,
                'byte_count': 0,
                'flow_count': 0,
                'last_update': time.time(),

                'prev_packet_count': 0,
                'prev_byte_count': 0,

                'throughput_mbps': 0.0,
                'avg_latency_ms': 0.0,
                'packet_loss_rate': 0.0,

                'tx_packets': 0,
                'tx_bytes': 0,

                'last_tx_packets_snapshot': None,
                'last_tx_bytes_snapshot': None
            }
        }

        self.qos_config = {
            'URLLC': {
                'bandwidth_percent': 50.0,
                'min_rate_mbps': 10.0,
                'max_rate_mbps': 50.0
            },
            'mMTC': {
                'bandwidth_percent': 50.0,
                'min_rate_mbps': 1.0,
                'max_rate_mbps': 50.0
            }
        }

        self.monitor_thread = hub.spawn(self._monitor_loop)
        wsgi = kwargs['wsgi']
        wsgi.register(SlicingRestController, {slice_api_instance_name: self})

        self.logger.info('=' * 70)
        self.logger.info('Slicing Switch 13 started (QUEUE-STATS)')
        self.logger.info('REST API: http://localhost:8080%s', url_base)
        self.logger.info('=' * 70)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info('Switch connected: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info('Switch disconnected: %016x', datapath.id)
                del self.datapaths[datapath.id]

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.logger.info('Config switch %016x', datapath.id)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id, priority=priority,
                                    match=match, instructions=inst, idle_timeout=idle_timeout,
                                    hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                    instructions=inst, idle_timeout=idle_timeout, hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    def _classify_packet(self, pkt):
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if not ip_pkt:
            return None

        dscp = (ip_pkt.tos >> 2) & 0x3F
        for slice_name, config in SLICE_CONFIG.items():
            if dscp in config['dscp']:
                return slice_name

        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)
        port = None
        if tcp_pkt:
            port = tcp_pkt.dst_port
        elif udp_pkt:
            port = udp_pkt.dst_port

        if port:
            for slice_name, config in SLICE_CONFIG.items():
                if port in config['ports']:
                    return slice_name
        return None

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        slice_type = self._classify_packet(pkt)
        actions = [parser.OFPActionOutput(out_port)]

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        udp_pkt = pkt.get_protocol(udp.udp)
        tcp_pkt = pkt.get_protocol(tcp.tcp)

        if slice_type and out_port != ofproto.OFPP_FLOOD:
            slice_config = SLICE_CONFIG[slice_type]
            queue_id = slice_config['queue_id']
            priority = slice_config['priority']

            actions.insert(0, parser.OFPActionSetQueue(queue_id))

            if ip_pkt and udp_pkt:
                match = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=ether_types.ETH_TYPE_IP,
                    ip_proto=17,
                    udp_dst=udp_pkt.dst_port,
                    ipv4_src=ip_pkt.src,
                    ipv4_dst=ip_pkt.dst
                )
            elif ip_pkt and tcp_pkt:
                match = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=ether_types.ETH_TYPE_IP,
                    ip_proto=6,
                    tcp_dst=tcp_pkt.dst_port,
                    ipv4_src=ip_pkt.src,
                    ipv4_dst=ip_pkt.dst
                )
            else:
                match = parser.OFPMatch(in_port=in_port, eth_src=src, eth_dst=dst)

            self.add_flow(datapath, priority, match, actions, idle_timeout=30, hard_timeout=60)
            self.slice_stats[slice_type]['flow_count'] += 1
            self.logger.info('Flow installed: %s -> %s via %s (queue %d)', src, dst, slice_type, queue_id)

        elif out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_src=src, eth_dst=dst)
            self.add_flow(datapath, 1, match, actions, idle_timeout=10, hard_timeout=30)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def _monitor_loop(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(5)

    def _request_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPQueueStatsRequest(datapath, 0, ofproto.OFPP_ANY, ofproto.OFPQ_ALL)
        datapath.send_msg(req)

    def _calculate_metrics(self, slice_name):
        stats = self.slice_stats[slice_name]
        now = time.time()
        dt = now - stats.get('last_update', now)

        if dt < 0.1:
            return

        cur_bytes = int(stats.get('tx_bytes', 0))
        cur_pkts = int(stats.get('tx_packets', 0))

        if stats.get('last_tx_bytes_snapshot') is None:
            stats['last_tx_bytes_snapshot'] = cur_bytes
            stats['last_tx_packets_snapshot'] = cur_pkts
            stats['throughput_mbps'] = 0.0
            stats['avg_latency_ms'] = 1.0
            stats['packet_loss_rate'] = 0.0
            stats['last_update'] = now
            stats['packet_count'] = cur_pkts
            stats['byte_count'] = cur_bytes
            stats['prev_packet_count'] = cur_pkts
            stats['prev_byte_count'] = cur_bytes
            return

        prev_bytes = int(stats.get('last_tx_bytes_snapshot', 0))
        prev_pkts = int(stats.get('last_tx_packets_snapshot', 0))

        dbytes = cur_bytes - prev_bytes
        dpkts = cur_pkts - prev_pkts

        if dbytes < 0:
            dbytes = 0
        if dpkts < 0:
            dpkts = 0

        throughput_bps = (dbytes * 8.0) / dt
        stats['throughput_mbps'] = throughput_bps / 1000000.0

        stats['avg_latency_ms'] = 1.0
        stats['packet_loss_rate'] = 0.0

        stats['last_tx_bytes_snapshot'] = cur_bytes
        stats['last_tx_packets_snapshot'] = cur_pkts
        stats['last_update'] = now

        stats['packet_count'] = cur_pkts
        stats['byte_count'] = cur_bytes
        stats['prev_packet_count'] = cur_pkts
        stats['prev_byte_count'] = cur_bytes

    @set_ev_cls(ofp_event.EventOFPQueueStatsReply, MAIN_DISPATCHER)
    def _queue_stats_reply_handler(self, ev):
        body = ev.msg.body

        for slice_name in ['URLLC', 'mMTC']:
            self.slice_stats[slice_name]['tx_packets'] = 0
            self.slice_stats[slice_name]['tx_bytes'] = 0

        for stat in body:
            queue_id = stat.queue_id

            slice_name = None
            for name, config in SLICE_CONFIG.items():
                if config['queue_id'] == queue_id:
                    slice_name = name
                    break

            if slice_name:
                self.slice_stats[slice_name]['tx_packets'] += stat.tx_packets
                self.slice_stats[slice_name]['tx_bytes'] += stat.tx_bytes

        for slice_name in ['URLLC', 'mMTC']:
            self._calculate_metrics(slice_name)

    def update_qos(self, qos_params):
        try:
            total_percent = 0
            for slice_name in ['URLLC', 'mMTC']:
                if slice_name in qos_params:
                    total_percent += qos_params[slice_name].get('bandwidth_percent', 0)

            if abs(total_percent - 100.0) > 0.1:
                raise ValueError('Sum must be 100%')

            for slice_name, params in qos_params.items():
                if slice_name in self.qos_config:
                    self.qos_config[slice_name].update(params)

            self.logger.info('QoS updated: URLLC=%s%%, mMTC=%s%%',
                             self.qos_config['URLLC']['bandwidth_percent'],
                             self.qos_config['mMTC']['bandwidth_percent'])

            return {'status': 'success', 'config': self.qos_config}
        except Exception as e:
            self.logger.error('QoS error: %s', str(e))
            return {'status': 'error', 'message': str(e)}

    def get_metrics(self):
        return {
            'timestamp': time.time(),
            'slices': {
                'URLLC': {
                    'packet_count': self.slice_stats['URLLC']['packet_count'],
                    'byte_count': self.slice_stats['URLLC']['byte_count'],
                    'flow_count': self.slice_stats['URLLC']['flow_count'],
                    'throughput_mbps': self.slice_stats['URLLC']['throughput_mbps'],
                    'avg_latency_ms': self.slice_stats['URLLC']['avg_latency_ms'],
                    'packet_loss_rate': self.slice_stats['URLLC']['packet_loss_rate'],
                    'last_update': self.slice_stats['URLLC']['last_update']
                },
                'mMTC': {
                    'packet_count': self.slice_stats['mMTC']['packet_count'],
                    'byte_count': self.slice_stats['mMTC']['byte_count'],
                    'flow_count': self.slice_stats['mMTC']['flow_count'],
                    'throughput_mbps': self.slice_stats['mMTC']['throughput_mbps'],
                    'avg_latency_ms': self.slice_stats['mMTC']['avg_latency_ms'],
                    'packet_loss_rate': self.slice_stats['mMTC']['packet_loss_rate'],
                    'last_update': self.slice_stats['mMTC']['last_update']
                }
            },
            'qos_config': self.qos_config,
            'datapaths': {dpid: {'id': dpid, 'connected': True} for dpid in self.datapaths.keys()}
        }


class SlicingRestController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(SlicingRestController, self).__init__(req, link, data, **config)
        self.slicing_app = data[slice_api_instance_name]

    @route('qos', url_base + '/qos', methods=['POST'])
    def update_qos(self, req, **kwargs):
        try:
            qos_params = json.loads(req.body)
            result = self.slicing_app.update_qos(qos_params)
            body = json.dumps(result, indent=2)
            status = 200 if result['status'] == 'success' else 400
            return Response(status=status, content_type='application/json', body=body)
        except Exception as e:
            error_response = {'status': 'error', 'message': str(e)}
            return Response(status=500, content_type='application/json', body=json.dumps(error_response))

    @route('metrics', url_base + '/metrics', methods=['GET'])
    def get_metrics(self, req, **kwargs):
        try:
            metrics = self.slicing_app.get_metrics()
            body = json.dumps(metrics, indent=2)
            return Response(status=200, content_type='application/json', body=body)
        except Exception as e:
            error_response = {'status': 'error', 'message': str(e)}
            return Response(status=500, content_type='application/json', body=json.dumps(error_response))

    @route('status', url_base + '/status', methods=['GET'])
    def get_status(self, req, **kwargs):
        try:
            status = {
                'status': 'running',
                'datapaths_count': len(self.slicing_app.datapaths),
                'slices': list(SLICE_CONFIG.keys()),
                'api_version': '1.0'
            }
            body = json.dumps(status, indent=2)
            return Response(status=200, content_type='application/json', body=body)
        except Exception as e:
            error_response = {'status': 'error', 'message': str(e)}
            return Response(status=500, content_type='application/json', body=json.dumps(error_response))

