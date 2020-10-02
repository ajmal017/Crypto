from findiff import FinDiff 
import findiff
import numpy as np
import pandas as pd
from skimage.transform import hough_line, hough_line_peaks


class TrendLine():
    
    def __init__(self, df, errpct, col):
        
        self.col = col
        self.df = df
        self.scale = (self.df[self.col].max() - self.df[self.col].min()) / len(self.df)
        self.errpct = errpct # 0.5%
        self.fltpct = self.scale * self.errpct
        self.h = self.df[self.col].tolist()
        
        
    def get_extrema(self, isMin):
        
        mom, momacc = self.momentum()
        return [x for x in range(len(mom))
        if (momacc[x] > 0 if isMin else momacc[x] < 0) and
          (mom[x] == 0 or # slope is 0
            (x != len(mom) - 1 and #check next day
              (mom[x] > 0 and mom[x+1] < 0 and
               self.h[x] >= self.h[x+1] or
               mom[x] < 0 and mom[x+1] > 0 and
               self.h[x] <= self.h[x+1]) or
             x != 0 and #check prior day
              (mom[x-1] > 0 and mom[x] < 0 and
               self.h[x-1] < self.h[x] or
               mom[x-1] < 0 and mom[x] > 0 and
               self.h[x-1] > self.h[x])))]  
    
    def get_bestfit(self, pts):
        
        xbar, ybar = [sum(x) / len (x) for x in zip(*pts)]

        def subcalc(x, y):
            tx, ty = x - xbar, y - ybar
            return tx * ty, tx * tx, x * x
        (xy, xs, xx) = [sum(q) for q in zip(*[subcalc(x, y) for x, y in pts])]
        m = xy / xs
        b = ybar - m * xbar
        ys = sum([np.square(y - (m * x + b)) for x, y in pts])
        ser = np.sqrt(ys / ((len(pts) - 2) * xs))
        return m, b, ys, ser, ser * np.sqrt(xx / len(pts))    

    def get_bestfit3(self, x0, y0, x1, y1, x2, y2):
    
        xbar, ybar = (x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3
        xb0, yb0, xb1, yb1, xb2, yb2 = x0-xbar, y0-ybar, x1-xbar, y1-ybar, x2-xbar, y2-ybar
        xs = xb0*xb0+xb1*xb1+xb2*xb2
        m = (xb0*yb0+xb1*yb1+xb2*yb2) / xs
        b = ybar - m * xbar
        ys0, ys1, ys2 = (y0 - (m * x0 + b)),(y1 - (m * x1 + b)),(y2 - (m * x2 + b))
        ys = ys0*ys0+ys1*ys1+ys2*ys2
        ser = np.sqrt(ys / xs)
        
        return m, b, ys, ser, ser * np.sqrt((x0*x0+x1*x1+x2*x2)/3)        
        
    def momentum(self):

        dx = 1 #1 day interval
        d_dx = FinDiff(0, dx, 1)
        d2_dx2 = FinDiff(0, dx, 2)
        clarr = np.asarray(self.df[self.col])
        mom = d_dx(clarr)
        momacc = d2_dx2(clarr)

        return mom, momacc



    def sorted_slope_trendln(self, Idxs):
        
        slopes, trend = [], []

        for x in range(len(Idxs)): #O(n^2*log n) algorithm
            slopes.append([])
            for y in range(x+1, len(Idxs)):
                slope = (self.h[Idxs[x]] - self.h[Idxs[y]]) / (Idxs[x] - Idxs[y])
                slopes[x].append((slope, y))

        for x in range(len(Idxs)):
            slopes[x].sort(key=lambda val: val[0])
            CurIdxs = [Idxs[x]]
            for y in range(0, len(slopes[x])):
                CurIdxs.append(Idxs[slopes[x][y][1]])
                if len(CurIdxs) < 3: continue
                res = self.get_bestfit([(p, self.h[p]) for p in CurIdxs])
                if res[3] <= self.fltpct:
                    CurIdxs.sort()
                    if len(CurIdxs) == 3:
                        trend.append((CurIdxs, res))
                        CurIdxs = list(CurIdxs)
                    else: CurIdxs, trend[-1] = list(CurIdxs), (CurIdxs, res)
                else: CurIdxs = [CurIdxs[0], CurIdxs[-1]] 

        return trend

    def naive_trendln(self,Idxs):
        trend = []
        
        for x in range(len(Idxs)):
            for y in range(x+1, len(Idxs)):
                for z in range(y+1, len(Idxs)):
                    trend.append(([Idxs[x], Idxs[y], Idxs[z]],
                                  self.get_bestfit3(Idxs[x], self.h[Idxs[x]],
                                               Idxs[y], self.h[Idxs[y]],
                                               Idxs[z], self.h[Idxs[z]])))
        return list(filter(lambda val: val[1][3] <= self.fltpct, trend))
    
    def get_trend_line(self, slope_support, intercept_support,
                             slope_resistance, intercept_resistance):

        x = np.array(range(len(self.df.index)))
        self.df['support'] = slope_support * x + intercept_support
        self.df['resistance'] = slope_resistance * x + intercept_resistance

        points = [(x[0],((self.df['resistance'].iloc[0]- self.df['support'].iloc[0])/2) + self.df['support'].iloc[0]),
                  (x[-1],((self.df['resistance'].iloc[-1] - self.df['support'].iloc[-1])/2) + self.df['support'].iloc[-1])]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords,np.ones(len(x_coords))]).T
        m, c = np.linalg.lstsq(A, y_coords)[0]
        #print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
        self.df['mid'] = m * x + c

        return self.df,m,c
    
    def trendln(self,mintrend,maxtrend):
        
        mom, momacc = self.momentum()
        self.df['Mom Acc'] = np.round(momacc,2)
        self.df['Mom'] = np.round(mom,2)

        coeff = findiff.coefficients(deriv=1, acc=1)

        slope_support = mintrend[-1][1][0]
        intercept_support = mintrend[-1][1][1]

        slope_resistance = maxtrend[-1][1][0]
        intercept_resistance = maxtrend[-1][1][1]

        df = self.get_trend_line(slope_support, intercept_support,
                                 slope_resistance, intercept_resistance)

        return  df    

    
## Hough Line Transform method


    def make_image(self,Idxs):
        
        max_size = int(np.ceil(2/np.tan(np.pi / (360 * 5)))) #~1146
        m, tested_angles = self.df[self.col].min(), np.linspace(-np.pi / 2, np.pi / 2, 360*5)
        height = int((self.df[self.col].max() - m + 0.01) * 100)
        mx = min(max_size, height)
        scl = 100.0 * mx / height
        image = np.zeros((mx, len(self.df))) #in rows, columns or y, x
        for x in Idxs:
            image[int((self.h[x] - m) * scl), x] = 255
        return image, tested_angles, scl, m
    
    
    def hough_points(self, pts, width, height, thetas):
        
        diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
        rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
        # Cache some resuable values
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        num_thetas = len(thetas)
        # Hough accumulator array of theta vs rho
        accumulator =np.zeros((2 * diag_len, num_thetas), dtype=np.int64)
        # Vote in the hough accumulator
        for i in range(len(pts)):
            x, y = pts[i]
            for t_idx in range(num_thetas):
              # Calculate rho. diag_len is added for a positive index
                rho=int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag_len
                accumulator[rho, t_idx] += 1
        return accumulator, thetas, rhos    
    
    
    def find_line_pts(self, Idxs, x0, y0, x1, y1):
        
        s = (y0 - y1) / (x0 - x1)
        i, dnm = y0 - s * x0, np.sqrt(1 + s*s)
        dist = [(np.abs(i+s*x-self.h[x])/dnm, x) for x in Idxs]
        dist.sort(key=lambda val: val[0])
        pts, res = [], None
        for x in range(len(dist)):
            pts.append((dist[x][1], self.h[dist[x][1]]))
            if len(pts) < 3: continue
            r = self.get_bestfit(pts)
            if r[3] > self.fltpct:
                pts = pts[:-1]
                break
            res = r
        pts = [x for x, _ in pts]
        pts.sort()
        return pts, res
    
    
    def houghpt_trendln(self, Idxs):
        
        max_size = int(np.ceil(2/np.tan(np.pi / (360 * 5)))) #~1146
        m, tested_angles = self.df[self.col].min(), np.linspace(-np.pi / 2, np.pi / 2, 360*5)
        height = int((self.df[self.col].max() - m + 1) * 100)
        mx = min(max_size, height)
        scl = 100.0 * mx / height
        acc, theta, d = self.hough_points([(x, int((self.h[x] - m) * scl)) for x in Idxs], mx, len(self.df), np.linspace(-np.pi / 2, np.pi / 2, 360*5))
        origin, lines = np.array((0, len(self.df))), []

        for x, y in np.argwhere(acc >= 3):
            dist, angle = d[x], theta[y]
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            y0, y1 = y0 / scl + m, y1 / scl + m
            pts, res = self.find_line_pts(Idxs, 0, y0, len(self.df), y1)
            if len(pts) >= 3: lines.append((pts, res))
        return lines
    
    def hough_trendln(self, Idxs):
        
        image, tested_angles, scl, m = self.make_image(Idxs)
        h, theta, d = hough_line(image, theta=tested_angles)
        origin, lines = np.array((0, image.shape[1])), []
        
        for pts, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=2)):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            y0, y1 = y0 / scl + m, y1 / scl + m
            pts, res = self.find_line_pts(Idxs, 0, y0, image.shape[1], y1)
            if len(pts) >= 3: lines.append((pts, res))

        return lines