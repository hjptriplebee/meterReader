import os
import json
from unittest import TestCase


class JsonModifier:
    """
    本类针对code cover testing 编写;本项目基于json config加载配置文件,很多识别算法需要调整参数开启一些算法的分支,进而让算法适用
    于另一种场景。在覆盖测试运行时,如果使用单一的配置文件,可能无法达到高代码覆盖率的目的,因而需要设计多个测试用例。
    设计多个测试用例的常用方案是:提供多个配置文件,让系统按需加载。然而这种方法冗余复杂,很多时候我们只需要改变config的一个参数,却需要
    提供另一个配置文件,其中的内容大部分跟之前是重复的.
    一种更好的方案是:在运行时使用脚本代码，动态修改仪表对应的配置文件,并及时写回（识别算法在Server进程中会读取该配置文件),该类提供了两个
    修改接口:modifyKv\modifyDic.根据输入的键值内容来修改配置文件
    同时,在测试完成后,该类能够自动回滚到默认的配置版本
    """
    revert_before_del = False  # 是否在实例销毁时恢复到备份
    src_config_path = ''  # config的路径
    dump_target_path = ''  # config备份的路径
    json_info = None  # json加载后的文件信息
    __src_file = None  # 打开的文件
    __target_dump_file = None  # 备份的文件
    changes = [{}]  # 保留每次对json的变更

    def __init__(self, meter_id, config_base_dir="/config", revert_before_del=False):
        self.src_config_path, self.dump_target_path = self.dumpConfig(meter_id, config_base_dir)
        self.__loadWritableJsonConfig()
        self.revert_before_del = revert_before_del

    def __del__(self):
        # 回滚并删除
        if self.revert_before_del:
            self.revertToOriginal()
            os.remove(self.dump_target_path)

    def __loadWritableJsonConfig(self):
        self.__src_file = open(self.src_config_path, "r")
        self.json_info = json.load(self.__src_file)

    def dumpConfig(self, meter_id, config_base_dir="/config"):
        """
        备份指定meterId的配置文件
        :param meter_id:
        :param config_base_dir:
        :return:
        """
        src_config_path = os.path.join(config_base_dir, meter_id + ".json")
        dump_target_path = os.path.join(config_base_dir, meter_id + "_dump.json")
        if os.path.isfile(src_config_path):
            self.__src_file = open(src_config_path, "rb")
            self.__target_dump_file = open(dump_target_path, "wb")
            # 备份原json config
            self.__target_dump_file.write(self.__src_file.read())
            self.__src_file.close()
            self.__target_dump_file.close()
        else:
            raise Exception("Configuration don't exist.")

        return src_config_path, dump_target_path

    def modifyKv(self, key, new_val):
        """
         根据k-v pair修改 json config
        :param key:
        :param new_val:
        :return:
        """
        if not self.__src_file.closed:
            self.__src_file.close()
        # 重写模式，将清楚原json config的全部数据
        self.__src_file = open(self.src_config_path, "w")
        if key in self.json_info:
            old_val = self.json_info[key]
            # 保存之前的修改值，方便回滚
            self.changes.append({key: old_val})
            self.json_info[key] = new_val
            self.__src_file.write(json.dumps(self.json_info))
            self.__src_file.close()
        else:
            self.__src_file.close()
            # 修改出错,根据备份回滚
            self.revertToOriginal()
            raise IOError("{} key don't exist in configuration.".format(key))

    def modifyDic(self, dic):
        """
        根据输入字典来修改json config
        :param dic:
        :return:
        """
        if not self.__src_file.closed:
            self.__src_file.close()
        self.__src_file = open(self.src_config_path, "w")
        for key, value in dic.items():
            if key in self.json_info:
                old_val = self.json_info[key]
                self.changes.append({key: old_val})
                self.json_info[key] = value
            else:
                self.__src_file.close()
                self.revertToOriginal()
                raise IOError("key : {} don't exist in configuration.".format(key))
        self.__src_file.write(json.dumps(self.json_info))
        self.__src_file.close()

    def revertToOriginal(self):
        """
        回滚到最初版本
        :return:
        """
        self.__target_dump_file = open(self.dump_target_path, "rb")
        self.__src_file = open(self.src_config_path, "wb")
        self.__src_file.truncate()
        self.__src_file.write(self.__target_dump_file.read())
        self.__src_file.close()

    def revert(self, backward=1):
        """
        回退修改backward步
        :param backward:
        :return:
        """
        self.__src_file = open(self.src_config_path, "w")
        for b in range(backward + 1):
            if len(self.changes) == 0:
                return
            pair = self.changes.pop()
            keys = pair.keys()
            for key in keys:
                self.json_info[key] = pair[key]
        self.__src_file.write(json.dumps(self.json_info))
        self.__src_file.close()
