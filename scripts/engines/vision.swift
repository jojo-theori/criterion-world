// macOS Vision 텍스트 영역 엔진.
//   recognize: VNRecognizeTextRequest. 줄 결과를 단어 단위 사각형(네 꼭짓점)으로 쪼갠다.
//   detect:    VNDetectTextRectanglesRequest. recognize 가 덮지 못한 영역만 보충한다.
// 커버마다 <outDir>/<stem>.json 을 쓰고, 이미 있으면 건너뛴다(--force 로 다시).
// 사용: swift scripts/engines/vision.swift <inDir> <outDir> [--force]

import Foundation
import Vision
import CoreGraphics
import ImageIO

let args = CommandLine.arguments.filter { $0 != "--force" }
let force = CommandLine.arguments.contains("--force")
let inDir = URL(fileURLWithPath: args[1])
let outDir = URL(fileURLWithPath: args[2])
let fm = FileManager.default
try fm.createDirectory(at: outDir, withIntermediateDirectories: true)

let coveredThreshold: CGFloat = 0.5

struct Region { let source: String; let text: String?; let confidence: Float; let quad: [CGPoint] }

func bounds(_ q: [CGPoint]) -> CGRect {
    let xs = q.map(\.x), ys = q.map(\.y)
    return CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
}
func quad(_ r: VNRectangleObservation) -> [CGPoint] { [r.topLeft, r.topRight, r.bottomRight, r.bottomLeft] }

func wordRegions(_ o: VNRecognizedTextObservation) -> [Region] {
    guard let c = o.topCandidates(1).first else { return [] }
    let s = c.string
    var out: [Region] = []
    var idx = s.startIndex
    while idx < s.endIndex {
        guard let start = s[idx...].firstIndex(where: { !$0.isWhitespace }) else { break }
        let end = s[start...].firstIndex(where: { $0.isWhitespace }) ?? s.endIndex
        if let box = try? c.boundingBox(for: start..<end) {
            out.append(Region(source: "recognize", text: String(s[start..<end]), confidence: c.confidence, quad: quad(box)))
        }
        idx = end
    }
    return out.isEmpty ? [Region(source: "recognize", text: s, confidence: c.confidence, quad: quad(o))] : out
}

func coveredRatio(_ a: CGRect, by bs: [CGRect]) -> CGFloat {
    let area = a.width * a.height
    guard area > 0 else { return 1 }
    let inter = bs.reduce(CGFloat(0)) { acc, b in let i = a.intersection(b); return acc + (i.isNull ? 0 : i.width * i.height) }
    return min(inter / area, 1)
}

let files = try fm.contentsOfDirectory(at: inDir, includingPropertiesForKeys: nil)
    .filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

var done = 0, skipped = 0
for file in files {
    let stem = file.deletingPathExtension().lastPathComponent
    let dest = outDir.appendingPathComponent(stem + ".json")
    if !force && fm.fileExists(atPath: dest.path) { skipped += 1; continue }
    guard let src = CGImageSourceCreateWithURL(file as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
    let w = img.width, h = img.height

    let t0 = Date()
    let rec = VNRecognizeTextRequest()
    rec.recognitionLevel = .accurate
    rec.usesLanguageCorrection = false
    rec.minimumTextHeight = 0.005
    let det = VNDetectTextRectanglesRequest()
    try VNImageRequestHandler(cgImage: img, options: [:]).perform([rec, det])

    let recRegions = (rec.results ?? []).flatMap(wordRegions)
    let recBounds = recRegions.map { bounds($0.quad) }
    let detRegions = (det.results ?? [])
        .map { Region(source: "detect", text: nil, confidence: $0.confidence, quad: quad($0)) }
        .filter { coveredRatio(bounds($0.quad), by: recBounds) < coveredThreshold }
    let elapsed = Date().timeIntervalSince(t0) * 1000

    // 픽셀 좌표, 원점 왼쪽 위
    func px(_ p: CGPoint) -> [Double] { [Double(p.x) * Double(w), (1 - Double(p.y)) * Double(h)] }
    let items: [[String: Any]] = (recRegions + detRegions).map { r in
        var d: [String: Any] = ["source": r.source, "confidence": Double(r.confidence), "poly": r.quad.map(px)]
        if let t = r.text { d["text"] = t }
        return d
    }
    let json: [String: Any] = ["file": file.lastPathComponent, "engine": "vision", "width": w, "height": h,
                               "elapsed_ms": Int(elapsed), "regions": items]
    try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]).write(to: dest)
    done += 1
}
print("vision: \(done) processed, \(skipped) skipped -> \(outDir.path)")
