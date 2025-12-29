#!/usr/bin/python3
# -*- coding: utf-8 -*-

# python 3.3.2+ drone.py Dos Script v.1
# by Can Yalçın
# only for legal purpose
"""
إطار عمل تعليمي للكشف عن إشارات RF غير معتادة
تحذير: هذا نظام تعليمي للتدريب فقط، ليس نظام كشف حقيقي عن الدرون
مطلوب: RTL-SDR جهاز
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class SignalType(Enum):
    """أنواع الإشارات المعروفة (لأغراض التعليم)"""
    UNKNOWN = "unknown"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"
    ISM_433 = "ism_433"
    ISM_868 = "ism_868"
    ISM_915 = "ism_915"
    CUSTOM = "custom"
    
@dataclass
class SignalDetection:
    """فئة تمثل اكتشاف إشارة"""
    timestamp: str
    frequency: float  # MHz
    bandwidth: float  # kHz
    power: float  # dBm
    signal_type: SignalType
    confidence: float  # 0-1
    location: Tuple[float, float]  # lat, lon
    signature: str  # توقيع مميز للإشارة

class EducationalRFDetector:
    """
    نظام تعليمي لتحليل إشارات RF باستخدام RTL-SDR
    لأغراض التدريب والبحث المشروع فقط
    """
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.known_signatures = self.load_known_signatures()
        self.detections_history: List[SignalDetection] = []
        self.alerts: List[Dict] = []
        
    def load_config(self, config_path: Optional[str]) -> Dict:
        """تحميل إعدادات النظام التعليمي"""
        default_config = {
            "frequency_ranges": {
                "ISM_433": (433.05, 434.79),  # MHz
                "ISM_868": (868.0, 868.6),
                "ISM_915": (902.0, 928.0),
                "WIFI_2G": (2400.0, 2483.5),
                "WIFI_5G": (5150.0, 5850.0),
                "BLUETOOTH": (2402.0, 2480.0)
            },
            "detection_threshold": -70,  # dBm
            "scan_interval": 1.0,  # seconds
            "location": (33.3152, 44.3661),  # بغداد
            "max_history": 1000
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except:
                print("⚠️  استخدام الإعدادات الافتراضية")
        
        return default_config
    
    def load_known_signatures(self) -> Dict:
        """تحميل توقيعات إشارات معروفة (لأغراض التعليم)"""
        # هذه توقيعات تعليمية وليست حقيقية
        return {
            "EDU_WIFI_BEACON": {
                "frequency_range": (2412, 2472),
                "bandwidth": 20,
                "pattern": "periodic_beacon",
                "type": SignalType.WIFI
            },
            "EDU_BT_ADVERT": {
                "frequency_range": (2402, 2480),
                "bandwidth": 2,
                "pattern": "frequency_hopping",
                "type": SignalType.BLUETOOTH
            },
            "EDU_ISM_CONTROL": {
                "frequency_range": (433.05, 434.79),
                "bandwidth": 0.1,
                "pattern": "control_signal",
                "type": SignalType.ISM_433
            }
        }
    
    def simulate_rtl_sdr_scan(self) -> List[Dict]:
        """
        محاكاة مسح RTL-SDR (للتعليم فقط)
        في النظام الحقيقي، سيتم استبدال هذا بـ pyrtlsdr
        """
        simulations = []
        
        # إضافة بعض الإشارات العشوائية للمحاكاة
        for _ in range(np.random.randint(1, 5)):
            freq_range = np.random.choice(list(self.config["frequency_ranges"].values()))
            freq = np.random.uniform(freq_range[0], freq_range[1])
            
            simulation = {
                "frequency": freq,
                "power": np.random.uniform(-90, -30),
                "bandwidth": np.random.uniform(0.1, 20),
                "samples": np.random.randn(1024) + 1j * np.random.randn(1024),
                "timestamp": datetime.now().isoformat()
            }
            simulations.append(simulation)
        
        return simulations
    
    def analyze_signal_characteristics(self, signal_data: Dict) -> Dict:
        """تحليل خصائص الإشارة (تعليمي)"""
        samples = signal_data.get("samples", np.array([]))
        
        if len(samples) == 0:
            return {"error": "لا توجد عينات"}
        
        # حساب خصائص الإشارة (تعليمي)
        power_spectrum = np.abs(np.fft.fft(samples))**2
        frequencies = np.fft.fftfreq(len(samples), 1/2.4e6)
        
        characteristics = {
            "peak_frequency": float(np.abs(frequencies[np.argmax(power_spectrum)]) / 1e6),
            "total_power": float(10 * np.log10(np.mean(power_spectrum) + 1e-10)),
            "bandwidth_estimate": float(np.sum(power_spectrum > 0.5 * np.max(power_spectrum)) * 2.4e6 / len(samples) / 1e3),
            "spectral_flatness": float(np.exp(np.mean(np.log(power_spectrum + 1e-10))) / np.mean(power_spectrum)),
            "modulation_score": np.random.random()  # محاكاة
        }
        
        return characteristics
    
    def classify_signal(self, characteristics: Dict) -> Tuple[SignalType, float]:
        """تصنيف الإشارة (خوارزمية تعليمية)"""
        
        freq = characteristics.get("peak_frequency", 0)
        bandwidth = characteristics.get("bandwidth_estimate", 0)
        
        # قواعد تصنيف تعليمية
        if 2400 <= freq <= 2483.5:
            if 20 <= bandwidth <= 40:
                return SignalType.WIFI, 0.8
            elif bandwidth < 2:
                return SignalType.BLUETOOTH, 0.7
        
        elif 433 <= freq <= 434.79:
            return SignalType.ISM_433, 0.6
        
        elif 868 <= freq <= 868.6:
            return SignalType.ISM_868, 0.6
        
        elif 902 <= freq <= 928:
            return SignalType.ISM_915, 0.6
        
        return SignalType.UNKNOWN, 0.3
    
    def detect_anomalies(self, signal_data: Dict, characteristics: Dict) -> Optional[Dict]:
        """اكتشاف إشارات غير عادية (لأغراض التدريب)"""
        
        anomalies = []
        
        # 1. تحقق من الترددات غير المصرح بها
        freq = characteristics.get("peak_frequency", 0)
        in_known_band = False
        
        for band_name, (f_low, f_high) in self.config["frequency_ranges"].items():
            if f_low <= freq <= f_high:
                in_known_band = True
                break
        
        if not in_known_band:
            anomalies.append({
                "type": "UNKNOWN_FREQUENCY",
                "severity": "MEDIUM",
                "message": f"إشارة على تردد غير معتاد: {freq:.2f} MHz"
            })
        
        # 2. تحقق من قوة الإشارة العالية
        power = characteristics.get("total_power", -100)
        if power > self.config["detection_threshold"]:
            anomalies.append({
                "type": "HIGH_POWER_SIGNAL",
                "severity": "LOW",
                "message": f"إشارة عالية الطاقة: {power:.1f} dBm"
            })
        
        # 3. تحقق من عرض النطاق غير المعتاد
        bandwidth = characteristics.get("bandwidth_estimate", 0)
        if bandwidth > 50:  # kHz
            anomalies.append({
                "type": "WIDE_BANDWIDTH",
                "severity": "MEDIUM",
                "message": f"عرض نطاق غير معتاد: {bandwidth:.1f} kHz"
            })
        
        return anomalies if anomalies else None
    
    def generate_signal_signature(self, signal_data: Dict) -> str:
        """إنشاء توقيع فريد للإشارة (تعليمي)"""
        import hashlib
        
        freq = signal_data.get("frequency", 0)
        power = signal_data.get("power", 0)
        timestamp = signal_data.get("timestamp", "")
        
        # إنشاء توقيع مبسط (في النظام الحقيقي يكون أكثر تعقيداً)
        signature_str = f"{freq:.3f}_{power:.1f}_{timestamp}"
        signature_hash = hashlib.md5(signature_str.encode()).hexdigest()[:8]
        
        return f"SIG_{signature_hash}"
    
    def scan_and_analyze(self) -> List[SignalDetection]:
        """تنفيذ دورة مسح وتحليل كاملة"""
        print(f"\n{'='*60}")
        print(f"جولة مسح RF - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        detections = []
        
        # محاكاة المسح (في النظام الحقيقي: sdr.read_samples())
        scanned_signals = self.simulate_rtl_sdr_scan()
        
        for i, signal in enumerate(scanned_signals):
            print(f"\n🔍 تحليل الإشارة #{i+1}")
            print(f"   التردد: {signal['frequency']:.2f} MHz")
            print(f"   القوة: {signal['power']:.1f} dBm")
            
            # تحليل الخصائص
            characteristics = self.analyze_signal_characteristics(signal)
            
            # تصنيف الإشارة
            signal_type, confidence = self.classify_signal(characteristics)
            print(f"   النوع: {signal_type.value} (ثقة: {confidence:.1%})")
            
            # اكتشاف الشذوذ
            anomalies = self.detect_anomalies(signal, characteristics)
            
            if anomalies:
                print(f"   ⚠️  تم اكتشاف {len(anomalies)} شذوذ:")
                for anomaly in anomalies:
                    print(f"      - {anomaly['message']}")
                    self.alerts.append({
                        **anomaly,
                        "frequency": signal['frequency'],
                        "timestamp": signal['timestamp']
                    })
            
            # إنشاء الاكتشاف
            detection = SignalDetection(
                timestamp=signal['timestamp'],
                frequency=signal['frequency'],
                bandwidth=characteristics.get('bandwidth_estimate', 0),
                power=signal['power'],
                signal_type=signal_type,
                confidence=confidence,
                location=self.config['location'],
                signature=self.generate_signal_signature(signal)
            )
            
            detections.append(detection)
            self.detections_history.append(detection)
            
            # الحفاظ على الحد الأقصى للسجل
            if len(self.detections_history) > self.config['max_history']:
                self.detections_history = self.detections_history[-self.config['max_history']:]
        
        return detections
    
    def generate_report(self, period_hours: int = 24) -> Dict:
        """توليد تقرير عن الفترة المحددة"""
        
        cutoff_time = datetime.now().timestamp() - (period_hours * 3600)
        
        recent_detections = [
            d for d in self.detections_history
            if datetime.fromisoformat(d.timestamp).timestamp() > cutoff_time
        ]
        
        recent_alerts = [
            a for a in self.alerts
            if datetime.fromisoformat(a['timestamp']).timestamp() > cutoff_time
        ]
        
        report = {
            "report_time": datetime.now().isoformat(),
            "period_hours": period_hours,
            "total_detections": len(recent_detections),
            "total_alerts": len(recent_alerts),
            "signal_type_distribution": {},
            "alerts_by_severity": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
            "frequency_coverage": {
                "known_bands": 0,
                "unknown_bands": 0
            },
            "top_anomalies": []
        }
        
        # توزيع أنواع الإشارات
        for detection in recent_detections:
            sig_type = detection.signal_type.value
            report["signal_type_distribution"][sig_type] = \
                report["signal_type_distribution"].get(sig_type, 0) + 1
            
            # تغطية الترددات
            in_known_band = False
            for f_low, f_high in self.config["frequency_ranges"].values():
                if f_low <= detection.frequency <= f_high:
                    in_known_band = True
                    break
            
            if in_known_band:
                report["frequency_coverage"]["known_bands"] += 1
            else:
                report["frequency_coverage"]["unknown_bands"] += 1
        
        # تحليل التنبيهات
        for alert in recent_alerts[-10:]:  # آخر 10 تنبيهات
            severity = alert.get("severity", "LOW")
            report["alerts_by_severity"][severity] += 1
            
            report["top_anomalies"].append({
                "time": alert['timestamp'],
                "type": alert['type'],
                "message": alert['message'],
                "frequency": alert.get('frequency', 0)
            })
        
        return report
    
    def run_continuous_monitoring(self, duration_minutes: int = 5):
        """تشغيل المراقبة المستمرة (للتدريب)"""
        import time
        
        print("\n" + "="*60)
        print("بدء المراقبة المستمرة للطيف الترددي")
        print(f"المدة: {duration_minutes} دقيقة")
        print("="*60 + "\n")
        
        start_time = time.time()
        scan_count = 0
        
        try:
            while time.time() - start_time < duration_minutes * 60:
                scan_count += 1
                print(f"\n📡 جولة المسح #{scan_count}")
                
                self.scan_and_analyze()
                
                # عرض ملخص كل 3 مسوحات
                if scan_count % 3 == 0:
                    report = self.generate_report(period_hours=1)
                    print("\n📊 ملخص سريع:")
                    print(f"   الإجمالي: {report['total_detections']} اكتشاف")
                    print(f"   التنبيهات: {report['total_alerts']}")
                    for severity, count in report['alerts_by_severity'].items():
                        if count > 0:
                            print(f"   {severity}: {count}")
                
                # الانتظار قبل المسح التالي
                time.sleep(self.config['scan_interval'])
                
        except KeyboardInterrupt:
            print("\n\n⏹️  توقف المراقبة بناءً على طلب المستخدم")
        
        # تقرير نهائي
        print("\n" + "="*60)
        print("تقرير المراقبة النهائي")
        print("="*60)
        
        final_report = self.generate_report(period_hours=24)
        
        print(f"\nالمسوحات المكتملة: {scan_count}")
        print(f"إجمالي الاكتشافات: {final_report['total_detections']}")
        print(f"إجمالي التنبيهات: {final_report['total_alerts']}")
        
        print("\nتوزيع أنواع الإشارات:")
        for sig_type, count in final_report['signal_type_distribution'].items():
            print(f"  {sig_type}: {count}")
        
        if final_report['top_anomalies']:
            print("\nأهم الشذوذات المكتشفة:")
            for anomaly in final_report['top_anomalies'][-5:]:
                print(f"  [{anomaly['time'][11:19]}] {anomaly['message']}")

# كود التكامل مع RTL-SDR الحقيقي (للتعليم)
class RealRTLSDRExtension:
    """
    مثال للاستخدام الفعلي لـ RTL-SDR (يتطلب pyrtlsdr)
    تحذير: للاستخدام المشروع فقط وفي بيئة مرخصة
    """
    
    @staticmethod
    def get_rtlsdr_usage_example():
        """إرجاع مثال لاستخدام RTL-SDR"""
        
        example_code = '''
from rtlsdr import RtlSdr
import numpy as np

class RTLSDRScanner:
    """ماسح RTL-SDR للاستخدام المشروع"""
    
    def __init__(self):
        self.sdr = RtlSdr()
        
    def configure_for_research(self):
        """تهيئة SDR لأغراض البحث المشروع"""
        # إعدادات لاستقبال الطقس NOAA (مثال مشروع)
        self.sdr.sample_rate = 2.4e6
        self.sdr.center_freq = 137.5e6  # تردد أقمار NOAA
        self.sdr.gain = 'auto'
        
    def scan_frequency(self, freq_hz, duration_sec=1):
        """مسح تردد محدد"""
        self.sdr.center_freq = freq_hz
        samples = self.sdr.read_samples(256*1024)
        
        return {
            'frequency': freq_hz / 1e6,
            'samples': samples,
            'power': 10 * np.log10(np.mean(np.abs(samples)**2))
        }
    
    def safe_shutdown(self):
        """إغلاق آمن للجهاز"""
        self.sdr.close()
'''
        
        return example_code

# الدالة الرئيسية
def main():
    """الدالة الرئيسية لتشغيل النظام التعليمي"""
    
    print("="*70)
    print("نظام كشف إشارات RF التعليمي - الإصدار التدريبي")
    print("="*70)
    print("\n⚠️  تحذير: هذا نظام تعليمي للتدريب فقط")
    print("   لا يمكن استخدامه للكشف عن الدرون الحقيقية")
    print("   لأغراض البحث والتعليم المشروع\n")
    
    detector = EducationalRFDetector()
    
    while True:
        print("\n" + "="*50)
        print("القائمة الرئيسية:")
        print("1. مسح ترددي واحد")
        print("2. مراقبة مستمرة (5 دقائق)")
        print("3. عرض التقرير")
        print("4. مثال استخدام RTL-SDR الحقيقي")
        print("5. الخروج")
        
        choice = input("\nاختر الخيار (1-5): ").strip()
        
        if choice == "1":
            detector.scan_and_analyze()
            
        elif choice == "2":
            detector.run_continuous_monitoring(duration_minutes=5)
            
        elif choice == "3":
            report = detector.generate_report(period_hours=24)
            print("\n📈 تقرير الـ24 ساعة الماضية:")
            print(json.dumps(report, indent=2, ensure_ascii=False))
            
        elif choice == "4":
            print("\n📡 مثال استخدام RTL-SDR الحقيقي:")
            print(RealRTLSDRExtension.get_rtlsdr_usage_example())
            print("\nملاحظة: يتطلب تثبيت pyrtlsdr")
            print("        واستخدام RTL-SDR جهاز فعلي")
            
        elif choice == "5":
            print("\nشكراً لاستخدام النظام التعليمي")
            print("التزم دائمًا بالقوانين واللوائح المحلية")
            break
            
        else:
            print("❌ خيار غير صالح")

if __name__ == "__main__":
    # إضافة ملف README افتراضي
    README = """
# نظام كشف إشارات RF التعليمي

## ⚠️ تحذيرات أمنية مهمة

1. **هذا نظام تعليمي فقط** للتدريب على مفاهيم استقبال وتحليل إشارات RF
2. **لا يمكن استخدامه للكشف عن الدرون** أو أي أنظمة طيران
3. **يتطلب التزامًا تامًا بالقوانين المحلية** والدولية للاتصالات
4. **ممنوع الاستخدام العسكري أو الأمني** دون تراخيص رسمية

## المتطلبات
